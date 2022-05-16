# -*- coding: utf-8 -*-
#=========================================================
# Beginning of electronutils.py
# @author: Tristan Britt
# @email: tristan.britt@mail.mcgill.ca
# @description: This file contains every util function 
# used by this package to read input from verbose NSCF 
# output file to read in electronic band structure
#
# This software is part of a package distributed under the 
# GPLv3 license, see LICENSE.txt
#=========================================================
from .structures import Band, symmetrize, extend_bragg
from .QEutils import hashabledict
from functools import lru_cache, partial
from crystals import Crystal
import multiprocessing as mp
from os import cpu_count
import re
import numpy as np

NCORES = cpu_count() - 1

def get_electronic_bands_nscf(file, decimate_core=0):
    with open(file, 'r') as fh:
        data = fh.read()

    num_bands = int(re.findall(r'number of Kohn-Sham states=(.*)', data)[-1].strip())
    recip_vecs_regex = re.findall( r'[b]\([1-3]\) = \(.*?\)', data)
    KS_regex = re.findall( r'[a-zA-Z] = \d{1,4}.\d{1,4} \d{1,4}.\d{1,4} \d{1,4}.\d{1,4}', data)

    recip_vecs = np.zeros((3,3))
    for id, vec in enumerate(recip_vecs_regex[:3]):
        recip_vecs[id,:] = list(map(float,(' '.join(vec.split(' ')).split()[3:6])))
    K_points = np.zeros((len(KS_regex),3))
    for id, k in enumerate(KS_regex):
        K_points[id,:] = np.linalg.inv(recip_vecs.T)@np.array(list(map(float,k.split(" ")[2:5])))

    bands = np.zeros((K_points.shape[0],num_bands))
    for id, _ in enumerate(K_points):
        try:
            test = data[data.find(KS_regex[id]):data.find(KS_regex[id+1])].replace(' ','\n')
        except:
            test = data[data.find(KS_regex[id]):].replace(' ','\n')
        vals = list()
        for match in re.finditer(r'\s*-?\d+\.\d+|(-?\d+\.\d{2}(?!\d))',test):
            s = match.start()
            e = match.end()
            vals.append(test[s:e].strip())
        bands[id,:] = np.array(list(map(float, vals[3:20])))
    return recip_vecs, K_points, bands[:,decimate_core:]

def prepare_bands(nscf_fname, band_ordering, reflections, decimate=3):
    """
    Prepare all bands for further calculations. Caching included.

    Parameters
    ----------
    reflections : iterable of 3-tuples
        Miller indices of the reflections to consider.
    decimate : int, optional
        Decimation number, i.e., only one in every `decimate` q-points will be kept. Increase this factor to
        lower the number of q-points considered in the calculations.
    temperatures : dict[str, float] or None, optional
        Band temperatures [K], e.g. {"LA": 100}. Default value is room temperature.

    Returns
    -------
    bands : dict[str, Band]
    """
    return _prepare_bands(nscf_fname, hashabledict(band_ordering), reflections=tuple(reflections), decimate=int(decimate))


@lru_cache(maxsize=16)
def _prepare_bands(nscf_fname, band_ordering, reflections, decimate=3):
    cryst = Crystal.from_pwscf(nscf_fname)
    _, k_points, frequencies = get_electronic_bands_nscf(nscf_fname)

    # NOTE
    # Optimization trick is to skip over a few k-points
    # This doesn't seem to mangle the output of this program
    # and it saves time.
    k_points = k_points[0::decimate, :]
    frequencies = frequencies[0::decimate, :]
    # Int -> Str for in-plane bands
    names = {v: k for k, v in band_ordering.items()}
    bands = [
        Band(
            name=names[band_index],
            q_points=k_points,
            frequencies=frequencies[:, band_index],
            crystal=cryst,
        )
        for band_index in range(frequencies.shape[1])
    ]
    with mp.Pool(NCORES) as pool:
        symmetrized_bands = pool.map(symmetrize, bands)
        extended_bands = pool.map(
            partial(extend_bragg, reflections=reflections), symmetrized_bands
        )
    return {band.name: band for band in extended_bands}
#=========================================================
# End of electronutils.py
#=========================================================