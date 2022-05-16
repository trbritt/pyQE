# -*- coding: utf-8 -*-
#=========================================================
# Beginning of structures.py
# @author: Tristan Britt
# @email: tristan.britt@mail.mcgill.ca
# @description: This file contains the necessary function-
# ality to take first-principles electronic band structure
# and tile it to the equivalent image that would be render-
# ed on a x-ray or electron detector.
#
#
# This software is part of a package distributed under the 
# GPLv3 license, see LICENSE.txt
#=========================================================

import numpy as np
from .QEutils import mapply, tile_over_rows, is_in_plane, roughly_unique_by_rows, change_of_basis

class Band:
    """
    Parameters
    ----------
    q_points : ndarray, shape (N, 3)
        Table of reciprocal space vectors where the band is defined.
    frequencies : ndarray, shape (N, 1)
        Band frequencies at every point in ``k_points`` [eV]
    crystal : crystals.Crystal instance
    hkls : ndarray, shape (N, 3), dtype int, optional
        Nearest Bragg-peak associated with each row in k_points.
        Default is the (000) reflection only.
    """

    def __init__(self, name, q_points, frequencies, crystal, hkls=None):
        if hkls is None:
            hkls = np.zeros_like(q_points)
        self.name = name
        self.q_points = q_points
        self.frequencies = frequencies
        self.crystal = crystal
        self.hkls = hkls

    def save(self, fname):
        """Save all band information"""
        np.savez(
            fname,
            name = self.name,
            q_points=self.q_points,
            frequencies=self.frequencies,
            hkls=self.hkls,
        )

    def k_points(self):
        """Determine the unique k-points in this band."""
        from_miller = change_of_basis(
            np.array(self.crystal.reciprocal_vectors), np.eye(3)
        )
        bragg = mapply(from_miller, self.hkls)
        return self.q_points - bragg

    def filter_gamma(self, radius):
        """Filter information so that k-points near Gamma are removed."""
        not_near_gamma = np.greater(np.linalg.norm(self.k_points(), axis=1), radius)

        return Band(
            name=self.name,
            q_points=self.q_points[not_near_gamma],
            frequencies=self.frequencies[not_near_gamma],
            crystal=self.crystal,
            hkls=self.hkls[not_near_gamma],
        )
    def apply_symops(self, kpoints, crystal, symprec=1e-1):
        """
        Apply symmetry operations to polarizations vectors and q-points

        kpoints : ndarray, shape (N,3)
            Scattering vector within one Brillouin zone
        crystal: crystals.Crystal
            Crystal object with the appropriate symmetry.
        symprec : float, optional
            Symmetry-determination precision.
        """
        # Change of basis matrices allow to express
        # transformations in other bases
        to_reciprocal = change_of_basis(
            np.array(crystal.lattice_vectors), np.array(crystal.reciprocal_vectors)
        )
        from_reciprocal = np.linalg.inv(to_reciprocal)
        reciprocal = lambda m: to_reciprocal @ m @ from_reciprocal

        transformed_k = list()

        transformations = crystal.symmetry_operations(symprec=symprec)
        in_plane_transformations = list(filter(is_in_plane, transformations))
        for transf in in_plane_transformations:
            rotation = transf[0:3, 0:3]
            transformed_k.append(mapply(reciprocal(rotation), kpoints))

        return np.vstack(transformed_k)
    



class Mode(Band):
    def __init__(self, name, q_points, frequencies, polarizations, crystal, hkls=None):
        super().__init__(name, q_points, frequencies, crystal, hkls)
        self.polarizations = polarizations

    def apply_symops(self, kpoints, polarizations, crystal, symprec=1e-1):
        """
        Apply symmetry operations to polarizations vectors and q-points

        kpoints : ndarray, shape (N,3)
            Scattering vector within one Brillouin zone
        polarizations : ndarray, shape (N, natoms, 3)
            Complex polarization vectors. Every row is associated with the corresponding row
            in `kpoints`.
        crystal: crystals.Crystal
            Crystal object with the appropriate symmetry.
        symprec : float, optional
            Symmetry-determination precision.
        """
        # Change of basis matrices allow to express
        # transformations in other bases

        to_frac = change_of_basis(np.eye(3), np.array(crystal.lattice_vectors))
        from_frac = np.linalg.inv(to_frac)
        cartesian = lambda m: from_frac @ m @ to_frac

        transformed_p = list()

        transformations = crystal.symmetry_operations(symprec=symprec)
        in_plane_transformations = list(filter(is_in_plane, transformations))
        for transf in in_plane_transformations:
            rotation = transf[0:3, 0:3]

            # Transforming polarizations is a little more complex
            # because of the extra dimension.
            newpols = np.copy(polarizations)
            for atm_index in range(len(crystal)):
                newpols[:, atm_index, :] = mapply(
                    cartesian(rotation), polarizations[:, atm_index, :]
                )

            transformed_p.append(newpols)
        
        return super().apply_symops(kpoints, crystal, symprec), np.vstack(transformed_p)


def symmetrize(mode):
    """
    Extend mode information by symmetrization.

    The input is assumed to be a Mode representing information in
    a single Brillouin zone (000), unsymmetrized.
    """
    assert np.allclose(mode.hkls, np.zeros_like(mode.q_points))

    k_points_frac = mode.q_points  # called k_points because only one brillouin zone
    frequencies = mode.frequencies
    cryst = mode.crystal
    # Conversion matrices
    to_fractional = change_of_basis(np.eye(3), np.array(cryst.reciprocal_vectors))
    from_fractional = np.linalg.inv(to_fractional)
    if isinstance(mode, Mode):
        polarizations = mode.polarizations

        # Apply symmetry information and tile arrays to same shape
        k_points_frac, polarizations = mode.apply_symops(
            k_points_frac, polarizations, crystal=cryst
        )
        k_points_frac, polarizations, frequencies = tile_over_rows(
            k_points_frac, polarizations, frequencies
        )
        # Change of basis to inverse angstroms
        # Called k_points because still inside a single Brillouin zone.
        k_points = mapply(from_fractional, k_points_frac)
        return Mode(
            name=mode.name,
            q_points=k_points,
            frequencies=frequencies,
            polarizations=polarizations,
            crystal=cryst,
            hkls=mode.hkls,
        )
    else:
        k_points_frac = mode.apply_symops(
            k_points_frac, crystal=cryst
        )
        k_points_frac, frequencies = tile_over_rows(
            k_points_frac, frequencies
        )
        k_points = mapply(from_fractional, k_points_frac)
        return Band(
            name=mode.name,
            q_points=k_points,
            frequencies=frequencies,
            crystal=cryst,
            hkls=mode.hkls,
        )

def decimate(mode, decimals=2):
    """
    Decimate the information contained in modes based on similarity
    i.e. q-points that are roughly the same will be purged.
    """

    # Filter qs to a slightly lower precision
    # Because of QE rounding error + symmetry operations,
    # lots of duplicated points...
    if isinstance(mode, Mode):
        q_points, polarizations, frequencies, hkls = roughly_unique_by_rows(
            mode.q_points,
            mode.polarizations,
            mode.frequencies,
            mode.hkls,
            decimals=decimals,
        )

        return Mode(
            name=mode.name,
            q_points=q_points,
            polarizations=polarizations,
            frequencies=frequencies,
            hkls=hkls,
            crystal=mode.crystal,
        )
    else:
        q_points, frequencies, hkls = roughly_unique_by_rows(
        mode.q_points,
        mode.frequencies,
        mode.hkls,
        decimals=decimals,
        )

        return Band(
            name=mode.name,
            q_points=q_points,
            frequencies=frequencies,
            hkls=hkls,
            crystal=mode.crystal,
        )

def extend_bragg(mode, reflections):
    """
    Expand mode information so that it covers the entire detector range.

    The input is assumed to be a Mode representing information in a single Brillouin zone (000).
    Symmetry operations are applied, as well as some filtering.
    """
    HANDLE_POL = False
    if isinstance(mode, Mode):
        HANDLE_POL = True
    q_points = mode.q_points
    if HANDLE_POL: polarizations = mode.polarizations
    frequencies = mode.frequencies

    over_reflections, hkls = list(), list()
    astar, bstar, cstar = mode.crystal.reciprocal_vectors
    for (h, k, l) in reflections:
        H = h * astar + k * bstar + l * cstar
        over_reflections.append(q_points + H[None, :])

        # Quick way to copy a chunk of h, k, l row-wise
        hkls.append(np.zeros_like(q_points) + np.array([h, k, l], dtype=int)[None, :])

    # Important to distinguish
    q_points = np.vstack(over_reflections)
    hkls = np.vstack(hkls)
    if HANDLE_POL:
        q_points, polarizations, frequencies, hkls = tile_over_rows(
            q_points, polarizations, frequencies, hkls
        )

        return Mode(
            name=mode.name,
            q_points=q_points,
            frequencies=frequencies,
            polarizations=polarizations,
            crystal=mode.crystal,
            hkls=hkls,
        )
    else:
        q_points, frequencies, hkls = tile_over_rows(
            q_points, frequencies, hkls
        )

        return Band(
            name=mode.name,
            q_points=q_points,
            frequencies=frequencies,
            crystal=mode.crystal,
            hkls=hkls,
        )
#=========================================================
# End of structures.py
#=========================================================