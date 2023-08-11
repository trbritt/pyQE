
# -*- coding: utf-8 -*-
#=========================================================
# Beginning of one_phonon.py
# @author: Tristan Britt
# @email: tristan.britt@mail.mcgill.ca
# @description: This file contains all functions to generate
# one phonon structure factors. Note these functions are 
# only dependent on the data, not the format, so they
# can be used with either JSON format.
#
# This software is part of a package distributed under the 
# GPLv3 license, see LICENSE.txt
#=========================================================
import numpy as np
import npstreams as ns
from scipy.constants import physical_constants
from .QEutils import coth, unique_by_rows, rowdot, change_of_basis, mapply
from .structures import Mode
from skued import affe
from itertools import islice

def ncells(crystal, side_length=2_500_000, depth=12.72):
    """Calculate the number of unit cells N present in a crystal of graphite,
    250um x 250um"""
    return int(side_length * side_length * depth / crystal.volume)

def phonon_amplitude(frequencies, temperature, broadening=None):
    """
    Phonon amplitude within the Debye-Waller factor

    Parameters
    ----------
    frequencies : ndarray, shape (N,)
        Frequencies of a single phonon mode [Hz].
    temperature : float
        Mode temperature [K].

    Returns
    -------
    amplitude : ndarray, shape (N,)
        Amplitude SQUARED

    References
    ----------
    Xu and Chiang (2005) Eq. 23
    """
    if broadening is not None:
        assert frequencies.shape[0] == broadening.size 
    # Factor of 1/N is calculated in the parent caller
    hbar = physical_constants["Planck constant over 2 pi in eV s"][0]
    kB = physical_constants["Boltzmann constant in eV/K"][0]
    occupancy = (hbar / frequencies) * coth(hbar * frequencies / (2 * kB * temperature))
    if broadening is None:
        return occupancy
    else:
        return occupancy * broadening if temperature > 300 else occupancy #only broaden hot modes 

def _debye_waller_factor(modes, temperatures, atm_index, broadening):
    """Calculate a Debye-Waller factor for one atom."""
    # This calculation assumes that the Debye-Waller factor is isotropic
    # i.e. it only depends on the magnitude of q and polarizations ek
    # The anisotropic factor is much more computationally expensive
    n = ncells(modes["LA"].crystal)
    # The sum happens for all q's, but the polarization vectors of a single
    # Brillouin zone. This is annoying to keep track of. Since polarization
    # vectors are the same across different Brillouin zones, we sum over all
    # zones. This leads to "double" counting, hence a correction factor based
    # on the number of Brilluoin zones `nzones`.
    hkls, *_ = unique_by_rows(modes["LA"].hkls)
    nzones = hkls.shape[0]
    
    def accumulator(mode):
        # This is the sum over k
        # Really, it is a sum over all q, and it will be corrected
        # in the parent function.
        temp = temperatures[mode.name]
        # if broadening is not None: #if we want to look at broadening, take difference with eq occupations
        #     amplitude = np.abs(phonon_amplitude(mode.frequencies, temp, broadening) - phonon_amplitude(mode.frequencies, temp))
        # else:
        #     amplitude = phonon_amplitude(mode.frequencies, temp, broadening) #broad is None, but still, for completeness
        amplitude = phonon_amplitude(mode.frequencies, temp, broadening)
        retval = (
            np.sum(
                (1 / n)
                * amplitude
                * np.linalg.norm(mode.polarizations[:, atm_index, :], axis=1) ** 2
            )
            / nzones
        )
        # print(f'Mode {mode.name} & Atom Indx {atm_index} : min {retval.min()}, max {retval.max()}')
        return retval
    # This is the sum over modes
    retval =  ns.sum(accumulator(m) for m in modes.values() )#if m.name != "ZA")
    # print(f'Atom Indx {atm_index} : {retval}')
    return retval


def debye_waller_factors(modes : dict[str, Mode], temperatures=None, broadening=None) -> list[np.ndarray]:
    """
    Compute the debye-waller factor based on all mode information.
    These modes are assumed to have been expanded, i.e. represent mode information
    over the entire detector range.

    For performance reasons, we consider the isotropic Debye-Waller effect.

    Parameters
    ----------
    modes : dict[str, Mode]
    temperatures : dict[str, float] or None, optional
        Mode temperatures [K]. Defaults to room temperature.

    Returns
    -------
    factors : iterable of narrays
        One factor for every atom in the unit cell.

    References
    ----------
    Xu and Chiang (2005) Eq. 19 (anisotropic) and Eq. 20 (isotropic)
    """
    amu_to_kg = 1.6605e-27  # atomic mass units to Kg

    if temperatures is None:
        temperatures = {m: 300 for m in modes.keys()}
    # We loop through atoms in order that they are visible in the PWSCF file
    # That's the tag properties on Atom objects
    atoms = sorted(modes["LA"].crystal, key=lambda a: a.tag)
    q2 = np.linalg.norm(modes["LA"].q_points, axis=1) ** 2
    prefactor = lambda atm: q2 / (12 * atm.mass * amu_to_kg)
    # Parallelizing this calculation is actually slower
    # The correction factor `nzones` accounts for the "double" counting
    # of multiple BZ in the sum
    tmp = [
        prefactor(atom) * _debye_waller_factor(modes, temperatures, index, broadening)
        for index, atom in enumerate(atoms)
    ]
    return tmp


def one_phonon_structure_factor(mode : Mode, dw_factors : list[np.ndarray]) -> np.ndarray:
    """
    Compute the one-phonon structure factor associated with a mode.

    Parameters
    ----------
    mode : Mode
        Mode defined at `N` q-points.
    dw_factors : iterable of ndarray, shapes (N,)
        Debye-Waller factors, at every q-point of `mode`, for each atom in the unit cell.

    Returns
    -------
    oneph: ndarray, shape (N,)
        One-phonon structure factor for `mode`.
    """
    qpoints, polarizations, hkls, crystal = (
        mode.q_points,
        mode.polarizations,
        mode.hkls,
        mode.crystal,
    )

    assert dw_factors[0].shape == (qpoints.shape[0],)
    assert qpoints.shape == hkls.shape
    assert polarizations[:, 0, :].shape == qpoints.shape

    q_norm = np.linalg.norm(qpoints, axis=1, keepdims=True)

    # We loop through atoms in order that they are visible in the PWSCF file
    atoms = sorted(crystal, key=lambda a: a.tag)
    accumulator = np.zeros(shape=(qpoints.shape[0], 1), dtype=complex)
    for atm_index, atm in enumerate(atoms):
        # Accumulator is built in pieces
        # because all of these arrays are pretty big
        arg = np.ones_like(
            accumulator, dtype=complex
        )  # because polarization are complex vectors

        arg *= np.exp(-1 * dw_factors[atm_index].reshape(-1, 1))
        arg *= affe(atm, q_norm) / np.sqrt(atm.mass)
        arg *= rowdot(qpoints, polarizations[:, atm_index, :])
        accumulator += arg

    return np.nan_to_num(accumulator)

def bragg_scattering(mode, reflections, dw_factors, cartesian_broadening=0.1):
    qpoints, hkls, crystal = (
        mode.q_points,
        mode.hkls,
        mode.crystal,
    )
    assert dw_factors[0].shape == (qpoints.shape[0],)
    assert qpoints.shape == hkls.shape

    atomic_positions = np.array([atm.coords_cartesian for atm in islice(sorted(crystal), len(crystal))])
    astar, bstar, cstar = crystal.reciprocal_vectors
    bragg_peaks = np.vstack(
        [h * astar + k * bstar + l * cstar for (h, k, l) in reflections]
    )
    q_norm = np.linalg.norm(qpoints, axis=1, keepdims=True)
    atoms = sorted(crystal, key=lambda a: a.tag)
    accumulator = np.zeros(shape=(qpoints.shape[0], 1), dtype=complex)
    from tqdm import tqdm
    for bragg in tqdm(bragg_peaks, total=bragg_peaks.shape[0], desc='Bragg intensity'):
        for atm_index, atm in enumerate(atoms):
            arg = np.ones_like(
                accumulator, dtype=complex
            )  
            arg *= np.exp(-1 * dw_factors[atm_index].reshape(-1, 1))
            arg *= affe(atm, q_norm)
            arg *= np.exp(-1j * rowdot(qpoints, atomic_positions[atm_index].reshape(-1,3)))
            arg *= np.prod(np.exp(-0.5*(qpoints - bragg)**2/cartesian_broadening**2) / (cartesian_broadening * np.sqrt(2*np.pi)))
            accumulator += arg
    return np.abs( np.nan_to_num(accumulator) )**2
#=========================================================
# End of one_phonon.py
#=========================================================