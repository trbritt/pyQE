from os import cpu_count
from pathlib import Path

import numpy as np
from crystals import Crystal

from scipy.interpolate import griddata
from skimage.filters import gaussian
from skued import detector_scattvectors
from tqdm import tqdm
from pyQE.electronutils import prepare_bands

BAND_ORDERING = {}
for i in range(17):
    BAND_ORDERING[i] = i

ALL_BANDS = sorted(set(BAND_ORDERING.keys()))
INPUT = Path("/home//trbritt//Desktop//MoS2//HT-MoS2//GGA//04_nscf") 
OUTPUT = INPUT / "symmetrised"
OUTPUT.mkdir(exist_ok=True)


NCORES = cpu_count() - 1
EPS = np.finfo(float).eps
# NCORES = 2
def render(band_str, reflections, smoothing_sigma=15):
    """
    Render the one-phonon structure factor map as visible on
    the Siwick research group detector, for a specific band.

    Parameters
    ----------
    band_str : str
        Band name, e.g. "LA"
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
        Band frequencies [meV]
    """
    reflections = tuple(reflections)  # need to be hashable for caching to work
    bands = prepare_bands(INPUT/'output.out', BAND_ORDERING, reflections)

    band = bands[band_str]

    # Create a grid of wavevectors that are visible on the detector
    # Also determine what reflections are visible on the detector
    qx, qy, _ = detector_scattvectors(
        keV=90,
        camera_length=0.25,
        shape=(2048, 2048),
        pixel_size=14e-6,
        center=(1024, 1024),
    )


    # Frequencies are required for the oneph-majority figure
    interpolated_f = griddata(
        points=band.q_points[:, 0:2],
        values=band.frequencies.reshape((-1, 1)),
        xi=(qx, qy),
        method="nearest",
        fill_value=0.0,
    )


    return qx, qy, np.squeeze(interpolated_f)


def calculate(band_str, reflections):
    """
    Plot the one-phonon structure factor map as visible on
    the Siwick research group detector, for a specific band,
    on a Matplotlib `Axes` object.

    Parameters
    ----------
    band_str : str
        Band name, e.g. "LA"
    reflections : iterable of 3-tuple
        Reflections to use in the render.
    """
    # Calculate the locations of all reflections
    # that we  can plot as scatter.
    # Note that this is different than the hkls arrays store in the Band class
    bands = prepare_bands(INPUT/'output.out', BAND_ORDERING, reflections)
    cryst = bands[0].crystal
    astar, bstar, cstar = cryst.reciprocal_vectors
    bragg_peaks = np.vstack(
        [h * astar + k * bstar + l * cstar for (h, k, l) in reflections]
    )

    qx, qy, f = render(band_str, reflections=reflections, smoothing_sigma=15)
    np.save(OUTPUT / f"{band_str}_freq.npy", f)
    np.save(OUTPUT / f"{band_str}_freq_smoothed.npy", gaussian(f, sigma=15))
    np.save(OUTPUT / f"qx.npy", qx)
    np.save(OUTPUT / f"qy.npy", qy)
    np.save(OUTPUT / "bragg_peaks.npy", bragg_peaks)


if __name__ == "__main__":
    in_plane_refls = filter(
        lambda tup: tup[2] == 0, Crystal.from_pwscf(INPUT/"output.out").bounded_reflections(12)
    )
    in_plane_refls = tuple(in_plane_refls)
    for band in tqdm(ALL_BANDS):
        calculate(band, in_plane_refls)
