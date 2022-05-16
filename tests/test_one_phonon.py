# -*- coding: utf-8 -*-
"""
Calculate the in-plane one-phonon structure factors for 1L-MoS2.

"""

from os import cpu_count
from pathlib import Path
import numpy as np
from crystals import Crystal
from scipy.interpolate import griddata
from skimage.filters import gaussian
from skued import detector_scattvectors
from tqdm import tqdm
from pyQE.phononutils import prepare_modes
from pyQE.one_phonon import debye_waller_factors, one_phonon_structure_factor

INPUT = Path("/home//trbritt//Desktop//MoS2//preliminary_data//one_phonon") 
OUTPUT = INPUT / "oneph_mos2"
OUTPUT.mkdir(exist_ok=True)

#according to you!
MODE_ORDERING = {
    "ZA": 0,
    "TA": 1,
    "LA": 2,
    "TO1": 3,
    "LO1": 4,
    "TO2": 5,
    "LO2": 6,
    "ZO1": 7,
    "ZO2": 8,
}

MODES = sorted(MODE_ORDERING.keys())
IN_PLANE_MODES = sorted(set(MODE_ORDERING.keys()) - {"ZA", "ZO1", "ZO2"})
ALL_MODES = sorted(set(MODE_ORDERING.keys()))
NCORES = cpu_count() - 1


def render(mode_str, reflections, smoothing_sigma=15):
    """
    Render the one-phonon structure factor map as visible on
    the Siwick research group detector, for a specific mode.

    Parameters
    ----------
    mode_str : str
        Mode name, e.g. "LA"
    reflections : iterable of 3-tuple
        Reflections to use in the render.
    smoothing_sigma : int, optional
        Size in pixel of the smoothing gaussian kernel.

    Returns
    -------
    image : ndarray, shape (2048, 2048)
        One-phonon structure factor amplitude squared
    qx, qy : ndarray, shape (2048, 2048)
        Q-point mesh on which the one-phonon structure factor was calculated.
    f : ndarray, shape (2048, 2048)
        Mode frequencies [meV]
    """
    reflections = tuple(reflections)  # need to be hashable for caching to work
    modes = prepare_modes(INPUT/"mos2.out", INPUT / "complete_Data_New_v3.json", MODE_ORDERING, reflections)
    Ms = debye_waller_factors(modes)

    mode = modes[mode_str]
    F1j = np.abs(one_phonon_structure_factor(mode, dw_factors=Ms)) ** 2

    # Create a grid of wavevectors that are visible on the detector
    # Also determine what reflections are visible on the detector
    qx, qy, _ = detector_scattvectors(
        keV=90,
        camera_length=0.25,
        shape=(2048, 2048),
        pixel_size=14e-6,
        center=(1024, 1024),
    )

    # Interpolation sucks
    # Here is an idea for further performance enhancements:
    #    https://stackoverflow.com/questions/20915502/speedup-scipy-griddata-for-multiple-interpolations-between-two-irregular-grids?noredirect=1
    #
    # Because the number of reciprocal space points is so large, we can do with only nearest interpolation
    interpolated_oneph = griddata(
        points=mode.q_points[:, 0:2],
        values=F1j,
        xi=(qx, qy),
        method="nearest",
        fill_value=0.0,
    )

    # Frequencies are required for the oneph-majority figure
    interpolated_f = griddata(
        points=mode.q_points[:, 0:2],
        values=mode.frequencies.reshape((-1, 1)),
        xi=(qx, qy),
        method="nearest",
        fill_value=0.0,
    )

    image = np.squeeze(interpolated_oneph)
    return gaussian(image, sigma=smoothing_sigma), qx, qy, np.squeeze(interpolated_f)


def calculate(mode_str, reflections):
    """
    Plot the one-phonon structure factor map as visible on
    the Siwick research group detector, for a specific mode,
    on a Matplotlib `Axes` object.

    Parameters
    ----------
    mode_str : str
        Mode name, e.g. "LA"
    reflections : iterable of 3-tuple
        Reflections to use in the render.
    """
    # Calculate the locations of all reflections
    # that we  can plot as scatter.
    # Note that this is different than the hkls arrays store in the Mode class
    modes = prepare_modes(
        INPUT/"mos2.out", INPUT / "complete_Data_New_v4.json", MODE_ORDERING, reflections
    )
    cryst = modes["LA"].crystal
    astar, bstar, cstar = cryst.reciprocal_vectors
    bragg_peaks = np.vstack(
        [h * astar + k * bstar + l * cstar for (h, k, l) in reflections]
    )

    image, qx, qy, f = render(mode_str, reflections=reflections, smoothing_sigma=15)
    np.save(OUTPUT / f"{mode_str}_oneph.npy", image)
    np.save(OUTPUT / f"{mode_str}_freq.npy", f)
    np.save(OUTPUT / f"{mode_str}_freq_smoothed.npy", gaussian(f, sigma=15))
    np.save(OUTPUT / f"qx.npy", qx)
    np.save(OUTPUT / f"qy.npy", qy)
    np.save(OUTPUT / "bragg_peaks.npy", bragg_peaks)


if __name__ == "__main__":
    in_plane_refls = filter(
        lambda tup: tup[2] == 0, Crystal.from_pwscf(INPUT/"mos2.out").bounded_reflections(12)
    )
    in_plane_refls = tuple(in_plane_refls)
    for mode in tqdm(ALL_MODES):
        calculate(mode, in_plane_refls)