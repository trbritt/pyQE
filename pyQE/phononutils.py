# -*- coding: utf-8 -*-
#=========================================================
# Beginning of phononutils.py
# @author: Tristan Britt
# @email: tristan.britt@mail.mcgill.ca
# @description: This file contains all util functions
# to take first-principles data from PH.x and matdyn.x
# and convert it to usable data that can be processed later
# for first principles calculations or for visualization
#
# This software is part of a package distributed under the 
# GPLv3 license, see LICENSE.txt
#=========================================================
import json
import numpy as np
import re
from crystals import Crystal
from crystals.parsers import PWSCFParser
from scipy.constants import physical_constants
import multiprocessing as mp
from functools import lru_cache, partial
from os import cpu_count, chdir, getcwd
import subprocess
from pathlib import Path
from .Kpoints import Kpoint_path, Bravais_lattice
from .QEutils import hashabledict, PwIn
from .structures import Mode, symmetrize, extend_bragg


#---------------------------------------------------------
# The section of files below contains the necessary funct-
# ionality to generate phonon data into JSON format #1 
# and the functions to load that data back to memory
# for calculations later of eg one-phonon structure factor
#---------------------------------------------------------

NCORES = cpu_count() - 1

def extract_info_ordered(fname, crystal, **kwargs):
    """
    Extract the information from JSON file

    Returns
    -------
    q_points : ndarray, shape (N, 3)
    frequencies: ndarray, shape (N, nmodes)
    polarizations: ndarray, shape (N, nmodes, natoms, 3)
    """
    # Reciprocal lattice vectors in units of 2pi/alat
    # so that the output q-points are indeed in "fractional coordinates"
    p = PWSCFParser(crystal.source)
    transf = np.linalg.inv(np.array(p.reciprocal_vectors_alat()))
    with open(fname, mode="r") as f:
        modes = json.load(f)


    q_points = []
    # eig_vector = []
    freq = []
    polarizations = np.zeros((len(modes['kpoints']),9,3,3), dtype=np.complex128)

    for nkpt in range(len(modes['kpoints'])):

        q_pt = np.array(
            [
                float(modes['kpoints'][nkpt][0]),
                float(modes['kpoints'][nkpt][1]),
                float(modes['kpoints'][nkpt][2]),
            ]
        )
        q_points.append(np.array(q_pt).dot(transf))

        frequencies = []

        for branch in range(9):

            frequency = float(modes['frequencies'][branch][nkpt])
            frequencies.append(np.array(frequency))

            for atom in range(3):
                polarizations[nkpt][branch][atom][0] = np.array(float(modes['polarizations real'][branch][nkpt][atom][0]) + float(modes['polarizations imaginary'][branch][nkpt][atom][0]) * 1j)
                polarizations[nkpt][branch][atom][1] = np.array(float(modes['polarizations real'][branch][nkpt][atom][1]) + float(modes['polarizations imaginary'][branch][nkpt][atom][1]) * 1j)
                polarizations[nkpt][branch][atom][2] = np.array(float(modes['polarizations real'][branch][nkpt][atom][2]) + float(modes['polarizations imaginary'][branch][nkpt][atom][2]) * 1j)


        # eig_vector.append(e_vec)
        freq.append(np.array(frequencies))

    q_points = np.asarray(q_points)
    # eig_vector = np.squeeze(np.asarray(eig_vector))
    
    speed_of_light_cm = (
        physical_constants["speed of light in vacuum"][0] * 100
    )  # in cm/s
    freq = np.asarray(freq) * speed_of_light_cm  # frequencies in Hz

    # Certain acoustic modes may have slightly negative frequencies
    # We shift frequencies up so that the minimum is always 0
    freq -= freq.min() + np.finfo(float).eps

    return q_points, freq, polarizations


def prepare_modes(crystal_fname, json_fname, modeordering, reflections, decimate=3):
    """
    Prepare all modes for further calculations. Caching included.

    Parameters
    ----------
    reflections : iterable of 3-tuples
        Miller indices of the reflections to consider.
    decimate : int, optional
        Decimation number, i.e., only one in every `decimate` q-points will be kept. Increase this factor to
        lower the number of q-points considered in the calculations.
    temperatures : dict[str, float] or None, optional
        Mode temperatures [K], e.g. {"LA": 100}. Default value is room temperature.

    Returns
    -------
    modes : dict[str, Mode]
    """

    return _prepare_modes(crystal_fname, json_fname, hashabledict(modeordering), reflections=tuple(reflections), decimate=int(decimate))


@lru_cache(maxsize=16)
def _prepare_modes(crystal_fname, json_fname, modeordering, reflections, decimate=3):
    cryst = Crystal.from_pwscf(crystal_fname)

    k_points, frequencies, polarizations = extract_info_ordered(
        json_fname, crystal=cryst
    )
    k_points = k_points[0::decimate, :]
    frequencies = frequencies[0::decimate, :]
    polarizations = polarizations[0::decimate, :]
    # Int -> Str for in-plane modes 
    names = {v: k for k, v in modeordering.items()}
    modes = [
        Mode(
            name=names[mode_index],
            q_points=k_points,
            frequencies=frequencies[:, mode_index],
            polarizations=polarizations[:, mode_index, :, :],
            crystal=cryst,
        )
        for mode_index in range(frequencies.shape[1])
    ]

    with mp.Pool(NCORES) as pool:
        symmetrized_modes = pool.map(symmetrize, modes)
        extended_modes = pool.map(
            partial(extend_bragg, reflections=reflections), symmetrized_modes
        )
    return {mode.name: mode for mode in extended_modes}

#=========================================================
# All above calculations require input of json file in
# _prepare modes. We now include routines to make this json
#=========================================================

def get_json_file(factor,path,params):
    c,Bravais=get_Input_v2(path)
    celldm_1=c.lattice_parameters[0]
    celldm_2=c.lattice_parameters[1]/c.lattice_parameters[0]
    celldm_3=c.lattice_parameters[2]/c.lattice_parameters[0]
    alpha=c.lattice_parameters[3]
    beta=c.lattice_parameters[4]
    gamma=c.lattice_parameters[5]
    Kpoints,data_Kpoint=Kpoint_path(Bravais,factor,celldm_1=celldm_1,celldm_2=celldm_2,celldm_3=celldm_3,alpha=alpha,beta=beta,gamma=gamma)
    path_band=params['Filename']+'{}_bands.json'.format(params['File'])
    data_Kpoint.append(params['Filename'])
    with open(path_band, 'w') as outfile:
        json.dump(data_Kpoint, outfile)
    return Kpoints

def get_Input_v2(path):
    c = Crystal.from_pwscf(path)

    Space_group = c.international_number
    Bravais_group = c.hm_symbol[0]
    ###Get Bravais lattice
    Bravais=Bravais_lattice(Space_group,Bravais_group)
    return c,Bravais

def getModes_matdyn(params, return_vals=False):
    """Get frequencies and eigenvectors (Polarizations) from matdyn.mode file""" 
    atoms = params['nmb_atoms']
    if params['Start']:
        file_modes = params['Filename'] + 'matdyn.modes'
        params.update({'Start': False})
    else:
        file_modes = params['Filename'] + 'matdyn_PP//matdyn.modes'

    with open(file_modes) as fil:
        data_fil = fil.readlines()

        kpt_len = int((atoms * 3 * (atoms + 1) + 5))
        kpt_nmb = int(len(data_fil) / kpt_len)

        k_points = np.zeros((kpt_nmb, 3))
        frequency = np.zeros((atoms * 3, kpt_nmb))
        polarization = np.zeros((atoms * 3, kpt_nmb, atoms, 3), dtype = np.complex128)

        for k in range(kpt_nmb):
            new_kpt = [float(i) for i in data_fil[k * kpt_len + 2].split()[2:]]
            k_points[k] = np.array(np.array(new_kpt))
            for f in range(3 * atoms):
                frequency[f][k] = float(data_fil[k * kpt_len + (atoms + 1) * f + 4].split()[-2])
                for at in range(atoms):
                    for i in range(3):
                        polarization[f][k][at][i] = float(data_fil[k * kpt_len + (atoms + 1) * f + at + 5].split()[i * 2 + 1]) + float(data_fil[k * kpt_len + (atoms + 1) * f + at + 5].split()[i * 2 + 2]) * 1j
   
    modes = [
        Mode(
            mode_index,
            k_points,
            frequency[mode_index, :],
            polarization[mode_index, :, :, :],
            None #Crystals object
        )
        for mode_index in range(frequency.shape[0])
    ]
    if return_vals:
        return k_points, frequency, polarization
    else:
        return modes, params

def perform_matdyn(startpoint, endpoint, params):
    """Write Matdyn Input file and compute modes along Path"""
    chdir(params['Filename'])
    Path(params['Filename']+'matdyn_PP').mkdir(exist_ok=True)
    chdir('matdyn_PP')

    path_new=getcwd()

    Input_file_name = '{}_path.matdyn.in'.format(params['File'])
    # input_file = '../../q2r/{}_881.fc'.format(params['File'])
    input_file = '../../q2r/HT-MoS2_PAW444.fc'
    output_file  = '{}_Gamma.freq'.format(params['File'])

    lines=[]
    lines.append('&INPUT')
    lines.append(' flfrc = "{}", ! input file containing the interatomic force constants in real space'.format(input_file))
    lines.append(' asr = "simple", ! Indicates type of Acoustic Sum Rules')
    lines.append(' flfrq = "{}", ! output file containing the frequency of q-modes'.format(output_file))
    lines.append(' eigen_similarity = .true.,')
    lines.append(' q_in_band_form = .true.')
    lines.append('/')
    lines.append('2')

    if params['Get_ends']:
        lines.append('{} {} {} {}'.format(*tuple(startpoint), (params['Length'] * 100)))
        params.update({'Get_ends' : False})
    else:
        lines.append('{} {} {} {}'.format(*tuple(startpoint), (params['points'] - 1)))
    lines.append('{} {} {} 0'.format(*tuple(endpoint)))

    path_in = '{}//{}'.format(path_new, Input_file_name)
    with open(path_in, "w", newline='\n') as g: #write file
        g.write(
            '\n'.join(lines)
        )
   
    output = subprocess.run('~//q-e-qe-6.5//bin//matdyn.x < {} > tmp.out'.format(Input_file_name), shell = True, check = True)
    # if output.returncode == 0:
    #     print('run')
    chdir("../")


def distance(kpt1, kpt2):
    return int(np.linalg.norm(kpt1 - kpt2) * 200)

def endpt(kpt1, kpt2, length, ind):
    return kpt1 + (kpt2 - kpt1) / length * ind

def get_start_end_points(modes, params):
    """Get the path along the area O, P1, P2, etc."""
    File_json = params['Filename'] + '{}_bands.json'.format(params['File'])
    with open(File_json) as fil:
        data_json = json.load(fil)

    Sym = [s.replace('$' , '') for s in data_json[0]]
    Path_Sym = params['Path']
   
    mode = modes[0]
    kpts = mode.q_points

    if params['Gamma']:
        startpoint = np.array([0.0, 0.0, 0.0])
        Start_Sym = ['Gamma']
    else:
        startpoint = np.zeros((1, 3))
        Start_Sym = [params['Start q']]

    path = [0]
    for length in data_json[1]:
        path.append(path[-1] + length + 1)

    kpoints = np.zeros((len(Path_Sym), 3));  Symbols = ["" for i in range(len(Path_Sym))]
    for ind in range(len(Sym)):
        if Sym[ind] in Path_Sym:
            for ind2 in range(len(Path_Sym)):
                if Path_Sym[ind2] == Sym[ind]:
                    Symbols[ind2] = Sym[ind]
                    kpoints[ind2] = kpts[path[ind]]
        if Sym[ind] in Start_Sym and Sym[ind] != ['Gamma']:
            startpoint = kpts[path[ind]]

    lengths = []
    for ind in range(1, len(kpoints)):
        lengths.append(distance(kpoints[ind-1], kpoints[ind]))

    endpoints=[kpoints[0]]
    for sec in range(len(lengths)):
        for ind in range(1, lengths[sec] + 1):
            endpoints.append(endpt(kpoints[sec], kpoints[sec + 1], lengths[sec], ind))
    return startpoint, endpoints


def order_modes(modes_0, modes_1, modes_ends, end, params):
    mod = params['nmb_atoms'] * 3

    freq = np.zeros((mod, mod))
    freq_end_0 = np.zeros((mod, mod))
    freq_end_1 = np.zeros((mod, mod))
    polar = np.zeros((mod, mod))
   
    for st1 in range(mod):
        for st2 in range(mod):
            #print(modes_0[st1].frequencies[-1], modes_ends[st2].frequencies[(end-1)*100])

            difference = np.linalg.norm(modes_0[st1].frequencies - modes_1[st2].frequencies)

            diff_polar = np.linalg.norm(modes_0[st1].polarizations - modes_1[st2].polarizations)

            freq[st1][st2] = difference / params['points'] * 100.
            polar[st1][st2] = diff_polar / params['points'] * 100.

            freq_end_0[st1][st2] = np.abs(modes_0[st1].frequencies[-1] - modes_ends[st2].frequencies[(end - 1) * 100])
            freq_end_1[st1][st2] = np.abs(modes_1[st1].frequencies[-1] - modes_ends[st2].frequencies[(end) * 100])

    get_order = []; counter = []; get_order_end_0 = []; get_order_end_1 = []
    for f in range(mod):
        counter.append(f)
        min_freq = np.min(freq[f])
        min_freq_end_0 = np.min(freq_end_0[f])
        min_freq_end_1 = np.min(freq_end_1[f])
        value = []; value_end_0 = []; value_end_1 = []

        for k in range(mod):
            if freq[f][k] - 10 < min_freq:
                value.append(k)
            if freq_end_0[f][k] == min_freq_end_0:
                value_end_0.append(k)
            if freq_end_1[f][k] == min_freq_end_1:
                value_end_1.append(k)
           
       
        if len(value) == 1:
            get_order.append(value[0])
        else:
            new_polar = []
            for k in value:
                new_polar.append(polar[f][k])

            min_polar = np.min(new_polar)
            for k in range(mod):
                if polar[f][k] == min_polar:
                    get_order.append(k)

        if len(value_end_0) == 1:
            get_order_end_0.append(value_end_0[0])
        if len(value_end_1) == 1:
            get_order_end_1.append(value_end_1[0])

    # print(get_order)
   
    if sorted(get_order) == sorted(counter):
       
        return get_order

    else:
        new_order = np.arange(mod)
        for f in range(mod):
            new_order[get_order_end_0[f]] = get_order_end_1[f]
         
        print("Differences of frequencies is too low to determine the path, using end points to order modes.")
        return new_order 


def computeJSON(params):

    modes, params = getModes_matdyn(params)
    _ = get_json_file(1000, '/home//trbritt//Desktop//MoS2//preliminary_data//one_phonon//mos2.out', params  )
    startpoint, endpoints = get_start_end_points(modes, params)
    params.update({'Length' : len(endpoints)})

    print("at the first ...")
    ###Get endpoints to compare """Future steps is to compare it to the ends to avoid errors when modes are crossing"""
    perform_matdyn(endpoints[0], endpoints[-1], params)
    modes_ends, params = getModes_matdyn(params)
    print("at the second ...")

    ###Perform interpolation between two points
    perform_matdyn(startpoint, endpoints[0], params)
    modes_0, params = getModes_matdyn(params)

    complete_freq = []; complete_polar = []; complete_qpoint = []
    from tqdm import trange
    for end in trange(1, len(endpoints)):

        perform_matdyn(startpoint, endpoints[end], params)
        modes_1, params = getModes_matdyn(params)

        freq_new = np.zeros((3 * params['nmb_atoms'], params['points'])); polar_new = np.zeros((3 * params['nmb_atoms'], params['points'], params['nmb_atoms'], 3), dtype = complex)
        for f in range(3 * params['nmb_atoms']):

            if len(complete_freq) < 3 * params['nmb_atoms']:
                freq2 = []; polar2 = []
                mod = f
            else:
                freq2 = complete_freq[f]; polar2 = complete_polar[f]
                mod = order[f]

            freq_new[f] = modes_1[mod].frequencies
            polar_new[f] = modes_1[mod].polarizations
        
            for k in range(len(modes_0[0].q_points)):
                if k % params['Factor'] == 0:
                    freq2.append(modes_0[mod].frequencies[k])
                    polar2.append(modes_0[mod].polarizations[k])
                    if f == 0:
                        complete_qpoint.append(modes_0[f].q_points[k])
        
            ###Include the last point
            if f == 0:
                complete_qpoint.append(modes_0[f].q_points[k])
            freq2.append(modes_0[mod].frequencies[k])
            polar2.append(modes_0[mod].polarizations[k])

            if len(complete_freq) < 3 * params['nmb_atoms']:
                complete_freq.append(freq2)
                complete_polar.append(polar2)

            else:
                complete_freq[f] = freq2
                complete_polar[f] = polar2

        order = order_modes(modes_0, modes_1, modes_ends, end, params)

        for f in range(3 * params['nmb_atoms']):
            modes_0[f].q_points = modes_1[f].q_points
            modes_0[f].frequencies = freq_new[f]; modes_0[f].polarizations = polar_new[f]


    ###Include the last line
    for f in range(3 * params['nmb_atoms']):
        mod = order[f]
        freq2 = complete_freq[f]
        polar2 = complete_polar[f]

        for k in range(len(modes_1[0].frequencies)):
            if k % params['Factor'] == 0:
                freq2.append(modes_0[mod].frequencies[k])
                polar2.append(modes_0[mod].polarizations[k])
                if f == 0:
                    complete_qpoint.append(modes_0[f].q_points[k])

        if f == 0:
            complete_qpoint.append(modes_0[f].q_points[k])
        freq2.append(modes_0[mod].frequencies[k])
        polar2.append(modes_0[mod].polarizations[k])

        complete_freq[f] = freq2
        complete_polar[f] = polar2

    # Plot_qpoints(complete_qpoint)
    complete_qpoint = np.array(complete_qpoint, dtype = float); complete_freq = np.array(complete_freq, dtype = float)
    complete_polar = np.array(complete_polar, dtype = complex)

    complete_polar_real = complete_polar.real.tolist(); complete_polar_imaginary = complete_polar.imag.tolist()
    return complete_freq, complete_qpoint, complete_polar_real, complete_polar_imaginary


#---------------------------------------------------------
# The section of files below contains the necessary funct-
# ionality to generate phonon data into JSON format #2 
# and the functions to load that data back to memory
# for primarily visualization later. The input format 
# requires more information be in a single location in 
# memory for performance reasons, and so I recognize that
# there is redundant functionality but for performance
# I've made a separate class that is capable of containing
# more information
# The functionality below is taken from 
# https://github.com/henriquemiranda/phononwebsite
#---------------------------------------------------------

class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.ndarray,np.number)):
            if np.iscomplexobj(obj):
                return [obj.real, obj.imag]
            else:
                return obj.tolist()
        return(json.JSONEncoder.default(self, obj))


#conversion variables
ang2bohr = 1.889725989
bohr_angstroem = 0.529177249
hartree_cm1 = 219474.63
eV = 27.211396132
Bohr = 1.88973

#from ase https://wiki.fysik.dtu.dk/ase/
chemical_symbols = ['X',  'H',  'He', 'Li', 'Be',
                    'B',  'C',  'N',  'O',  'F',
                    'Ne', 'Na', 'Mg', 'Al', 'Si',
                    'P',  'S',  'Cl', 'Ar', 'K',
                    'Ca', 'Sc', 'Ti', 'V',  'Cr',
                    'Mn', 'Fe', 'Co', 'Ni', 'Cu',
                    'Zn', 'Ga', 'Ge', 'As', 'Se',
                    'Br', 'Kr', 'Rb', 'Sr', 'Y',
                    'Zr', 'Nb', 'Mo', 'Tc', 'Ru',
                    'Rh', 'Pd', 'Ag', 'Cd', 'In',
                    'Sn', 'Sb', 'Te', 'I',  'Xe',
                    'Cs', 'Ba', 'La', 'Ce', 'Pr',
                    'Nd', 'Pm', 'Sm', 'Eu', 'Gd',
                    'Tb', 'Dy', 'Ho', 'Er', 'Tm',
                    'Yb', 'Lu', 'Hf', 'Ta', 'W',
                    'Re', 'Os', 'Ir', 'Pt', 'Au',
                    'Hg', 'Tl', 'Pb', 'Bi', 'Po',
                    'At', 'Rn', 'Fr', 'Ra', 'Ac',
                    'Th', 'Pa', 'U',  'Np', 'Pu',
                    'Am', 'Cm', 'Bk', 'Cf', 'Es',
                    'Fm', 'Md', 'No', 'Lr']

atomic_mass = [   None,      1.00794,    4.002602,     6.941,   9.012182,
                10.811,      12.0107,     14.0067,   15.9994, 18.9984032,
               20.1797,  22.98976928,      24.305,26.9815386,    28.0855,
             30.973762,       32.065,      35.453,    39.948,    39.0983,
                40.078,    44.955912,      47.867,   50.9415,    51.9961,
             54.938045,       55.845,   58.933195,   58.6934,     63.546,
                 65.38,       69.723,       72.64,   74.9216,      78.96,
                79.904,       83.798,     85.4678,     87.62,   88.90585,
                91.224,     92.90638,       95.96,      None,     101.07,
              102.9055,       106.42,    107.8682,   112.411,    114.818,
                118.71,       121.76,       127.6, 126.90447,    131.293,
           132.9054519,      137.327,   138.90547,   140.116,  140.90765,
               144.242,         None,      150.36,   151.964,     157.25,
             158.92535,         162.5,  164.93032,   167.259,  168.93421,
               173.054,      174.9668,     178.49, 180.94788,     183.84,
               186.207,        190.23,    192.217,   195.084, 196.966569,
                200.59,      204.3833,      207.2,  208.9804,       None,
                  None,          None,       None,      None,       None,
             232.03806,     231.03588,  238.02891,      None,       None,
                  None,          None,       None,      None,       None,
                  None,          None,       None,      None,       None,
                  None,          None,       None,      None,       None,
                  None,          None,       None,      None,       None,
                  None,          None,       None,      None]

atomic_numbers = {}
for Z, symbol in enumerate(chemical_symbols):
    atomic_numbers[symbol] = Z

import numpy as np

def red_car(red,lat):
    """
    Convert reduced coordinates to cartesian
    """
    return np.array([coord[0]*lat[0]+coord[1]*lat[1]+coord[2]*lat[2] for coord in red])

def car_red(car,lat):
    """
    Convert cartesian coordinates to reduced
    """
    return np.array([np.linalg.solve(np.array(lat).T,coord) for coord in car])

def rec_lat(lat):
    """
    Calculate the reciprocal lattice vectors
    """
    a1,a2,a3 = np.array(lat)
    v = np.dot(a1,np.cross(a2,a3))
    b1 = np.cross(a2,a3)/v
    b2 = np.cross(a3,a1)/v
    b3 = np.cross(a1,a2)/v
    return np.array([b1,b2,b3])


def estimate_band_connection(prev_eigvecs, eigvecs, prev_band_order):
    """ 
    A function to order the phonon eigenvectors taken from phonopy
    """
    metric = np.abs(np.dot(prev_eigvecs.conjugate().T, eigvecs))
    connection_order = []
    indices = list(range(len(metric)))
    indices.reverse()
    for overlaps in metric:
        maxval = 0
        for i in indices:
            val = overlaps[i]
            if i in connection_order:
                continue
            if val > maxval:
                maxval = val
                maxindex = i
        connection_order.append(maxindex)

    band_order = [connection_order[x] for x in prev_band_order]
    return band_order

class Phonon():
    """ 
    Class to hold and manipulate generic phonon dispersions data 
    output .json files to be read by the phononwebsite
    """
    def reorder_eigenvalues(self):
        """
        compare the eigenvectors that correspond to the different eigenvalues
        to re-order the eigenvalues and solve the band-crossings
        """
        #vector transformations
        dim = (self.nqpoints, self.nphons, self.nphons)
        vectors = self.eigenvectors.view(complex).reshape(dim)
        
        eig = np.zeros([self.nqpoints,self.nphons])
        eiv = np.zeros([self.nqpoints,self.nphons,self.nphons],dtype=complex)
        #set order at gamma
        order = list(range(self.nphons))
        eig[0] = self.eigenvalues[0]
        eiv[0] = vectors[0]
        for k in range(1,self.nqpoints):
            order = estimate_band_connection(vectors[k-1].T,vectors[k].T,order)
            for n,i in enumerate(order):
                eig[k,n] = self.eigenvalues[k,i]
                eiv[k,n] = vectors[k,i]
        
        #update teh eigenvales with the ordered version
        self.eigenvalues  = eig
        dim = (self.nqpoints,self.nphons,self.natoms,3,2)
        self.eigenvectors = eiv.view(float).reshape(dim)

    def get_chemical_formula(self):
        """
        from ase https://wiki.fysik.dtu.dk/ase/
        """
        numbers = self.atom_numbers
        elements = np.unique(numbers)
        symbols = np.array([chemical_symbols[e] for e in elements])
        counts = np.array([(numbers == e).sum() for e in elements])

        ind = symbols.argsort()
        symbols = symbols[ind]
        counts = counts[ind]

        if 'H' in symbols:
            i = np.arange(len(symbols))[symbols == 'H']
            symbols = np.insert(np.delete(symbols, i), 0, symbols[i])
            counts = np.insert(np.delete(counts, i), 0, counts[i])
        if 'C' in symbols:
            i = np.arange(len(symbols))[symbols == 'C']
            symbols = np.insert(np.delete(symbols, i), 0, symbols[i])
            counts = np.insert(np.delete(counts, i), 0, counts[i])

        formula = ''
        for s, c in zip(symbols, counts):
            formula += s
            if c > 1:
                formula += str(c)

        return formula
        #end from ase

    def save_compressed(self):
        """ 
        Save phonon data in a HDF5 file
        """
        import h5py

        natypes = len(self.chemical_symbols)
        str10 = np.dtype(('S10', 2))

        #save all this stuff on a netcdf file
        with h5py.File("system.h5", 'w') as fh:
            SYSTEM = fh.create_group('system')
            SYSTEM.attrs['complex'] = 2
            SYSTEM.attrs['number_of_cartesian_dimensions'] = 3
            SYSTEM.attrs['number_of_reduced_dimensions'] = 3
            SYSTEM.attrs['number_of_atom_species'] = natypes
            SYSTEM.attrs['number_of_qpoints'] = self.nqpoints
            SYSTEM.attrs['number_of_atoms'] = self.natoms
            SYSTEM.attrs['number_of_phonon_modes'] = self.nphons
            SYSTEM.attrs['symbol_length'] =2

            PHONONS = fh.create_group('vibrations')
            PHONONS.create_dataset('primitive_vectors', data = self.cell/bohr_angstroem)
            PHONONS.create_dataset('reduced_atom_positions', data=self.pos)
            PHONONS.create_dataset('chemical_symbols', data = np.array(["%2s"%a for a in self.chemical_symbols],dtype=str10))
            PHONONS.create_dataset('qpoints', data = np.array(self.qpoints))
            PHONONS.create_dataset('eigenvalues', data = self.eigenvalues)
            PHONONS.create_dataset('atypes', data = np.array([np.where(self.chemical_symbols == a) for a in self.atom_types]))
            PHONONS.create_dataset('eigendisplacements', data = self.eigenvectors)


    def get_distances_qpts(self):

        #calculate reciprocal lattice
        rec = rec_lat(self.cell)
        #calculate qpoints in the reciprocal lattice
        car_qpoints = red_car(self.qpoints,rec)

        self.distances = []
        distance = 0
        #iterate over qpoints
        for k in range(1,self.nqpoints):
            self.distances.append(distance);

            #calculate distances
            step = np.linalg.norm(car_qpoints[k]-car_qpoints[k-1])
            distance += step

        #add the last distances
        self.distances.append(distance)

    def get_highsym_qpts(self):
        """ 
        Iterate over all the qpoints and obtain the high symmetry points 
        as well as the distances between them
        """
       
        def collinear(a,b,c):
            """
            checkkk if three points are collinear
            """
            d = [[a[0],a[1],1],
                 [b[0],b[1],1],
                 [c[0],c[1],1]]
            return np.isclose(np.linalg.det(d),0,atol=1e-5)

        #iterate over qpoints
        qpoints = self.qpoints;
        self.highsym_qpts = [[0,'']]
        for k in range(1,self.nqpoints-1):
            #detect high symmetry qpoints
            if not collinear(qpoints[k-1],qpoints[k],qpoints[k+1]):
                self.highsym_qpts.append((k,''))
        #add final k-point
        self.highsym_qpts.append((self.nqpoints-1,''))
    
        #if the labels are defined, assign them to the detected kpoints
        if self.labels_qpts:
            nhiqpts = len(self.highsym_qpts)
            nlabels = len(self.labels_qpts)
            if nlabels == nhiqpts:
                #fill in the symbols
                self.highsym_qpts = [(q,s) for (q,l),s in zip(self.highsym_qpts,self.labels_qpts)] 
            else:
                raise ValueError("Wrong number of q-points specified. "
                                 "Found %d high symmetry qpoints but %d labels"%(nhiqpts,nlabels))
        else:
            print("The labels of the high symmetry k-points are not known. "
                  "They can be changed in the .json file manualy.") 
        return self.highsym_qpts

    def set_repetitions(self,reps):
        """
        Get the number of repetitions based from a string
        """
        if   ',' in reps: reps = reps.split(',')
        elif ' ' in reps: reps = reps.split(' ')
        self.reps = [int(r) for r in reps]       
        print(self.reps)

    def set_labels(self,labels):
        """
        Use a string to set the names of the k-points.
        There are two modes:
            1 the string is a list of caracters and each caracter corresponds to one k-point
            2 the string is a set of words comma or space separated.
        """
        if   ',' in labels: self.labels_qpts = labels.split(',')
        elif ' ' in labels: self.labels_qpts = labels.split(' ')
        else:               self.labels_qpts = labels

    def write_json(self,prefix=None,folder='.'):
        """
        Write a json file to be read by javascript
        """
        if prefix: name = prefix
        else:      name = self.name

        f = open("%s/%s.json"%(folder,name),"w")

        if self.highsym_qpts == None:
            self.get_highsym_qpts()

        red_pos = red_car(self.pos,self.cell)
        #create the datastructure to be put on the json file
        data = {"name":         self.name,             # name of the material on the website
                "natoms":       self.natoms,           # number of atoms
                "lattice":      self.cell,             # lattice vectors (bohr)
                "atom_types":   self.atom_types,       # atom type for each atom (string)
                "atom_numbers": self.atom_numbers,     # atom number for each atom (integer)
                "formula":      self.chemical_formula, # chemical formula
                "qpoints":      self.qpoints,          # list of point in the reciprocal space
                "repetitions":  self.reps,             # default value for the repetititions 
                "atom_pos_car": red_pos,               # atomic positions in cartesian coordinates
                "atom_pos_red": self.pos,              # atomic positions in reduced coordinates
                "eigenvalues":  self.eigenvalues,      # eigenvalues (in units of cm-1)
                "distances":    self.distances,        # list distances between the qpoints 
                "highsym_qpts": self.highsym_qpts,     # list of high symmetry qpoints
                "vectors":      self.eigenvectors}     # eigenvectors

        f.write(json.dumps(data,cls=JsonEncoder,indent=1))
        f.close()

    def open_json(self,prefix=None,folder='.',host='localhost',port=8000):
        """
        Create a json file and open it on the webbrowser
        
        1. Create a thread with HTTP server on localhost to provide the file to the page
        2. Open the webpage indicating it to open the file from the localhost
        3. Wait for user to kill HTTP server 
        (TODO: once the file is served the server can shutdown automatically)
        """

        if prefix: name = prefix
        else:      name = self.name
        filename = "%s/%s.json"%(folder,name)

        #deactivate for now
        #open_file_phononwebsite(filename,host=host,port=port)

    def __str__(self):
        text = ""
        text += "name: %s\n"%self.name
        text += "cell:\n"
        for i in range(3):
            text += ("%12.8lf "*3)%tuple(self.cell[i])+"\n"
        text += "atoms:\n"
        for a in range(self.natoms):
            atom_pos_string = "%3s %3d"%(self.atom_types[a],self.atom_numbers[a])
            atom_typ_string = ("%12.8lf "*3)%tuple(self.pos[a]) 
            text += atom_pos_string+atom_typ_string+"\n"
        text += "atypes:\n"
        for cs,an in zip(self.chemical_symbols,self.atomic_numbers):
            text += "%3s %d\n"%(cs,an)
        text += "chemical formula:\n"
        text += self.chemical_formula+"\n"
        text += "nqpoints:\n"
        text += str(self.nqpoints)
        text += "\n"
        return text


""" Read phonon dispersion from quantum espresso """
import numpy as np

class QePhonon(Phonon):
    """
    Class to read phonons from Quantum Espresso

    Input:
        prefix: <prefix>.scf file where the structure is stored
                <prefix>.modes file that is the output of the matdyn.x or dynmat.x programs
    """
    def __init__(self,prefix,name,reps=(3,3,3),folder='.',
                 highsym_qpts=None,reorder=True,scf=None,modes=None):
        self.prefix = prefix
        self.name = name
        self.reps = reps
        self.folder = folder
        self.highsym_qpts = highsym_qpts

        #read atoms
        if scf:   filename = "%s/%s"%(self.folder,scf)
        else :    filename = "%s/%s.scf"%(self.folder,self.prefix)
        self.read_atoms(filename)
        
        #read modes
        if modes: filename = "%s/%s"%(self.folder,modes)
        else :    filename = "%s/%s.modes"%(self.folder,self.prefix)
        self.read_modes(filename)
        

        #reorder eigenvalues
        if reorder:
            self.reorder_eigenvalues()
        self.get_distances_qpts()
        self.labels_qpts = None

    def read_modes(self,filename):
        """
        Function to read the eigenvalues and eigenvectors from Quantum Expresso
        """
        with open(filename[1:],'r') as f:
            file_list = f.readlines()
            file_str  = "".join(file_list)

        #determine the number of atoms
        mode_index_re = re.findall( '(?:freq|omega) ?\((.+)\)', file_str )
        if len(mode_index_re) < 1: raise ValueError("Failed reading number of phonon modes.")
        nphons = max([int(x) for x in mode_index_re])
        atoms = int(nphons/3)

        #check if the number fo atoms is the same
        if atoms != self.natoms:
            print("The number of atoms in the <>.scf file is not the same as in the <>.modes file")
            exit(1)

        #determine the number of qpoints
        self.nqpoints = len( re.findall('q = ', file_str ) )
        nqpoints = self.nqpoints

        eig = np.zeros([nqpoints,nphons])
        vec = np.zeros([nqpoints,nphons,atoms,3],dtype=complex)
        qpt = np.zeros([nqpoints,3])
        for k in range(nqpoints):
            #iterate over qpoints
            k_idx = 2 + k*((atoms+1)*nphons + 5)
            #read qpoint
            qpt[k] = list(map(float, file_list[k_idx].split()[2:]))
            for n in range(nphons):
                #read eigenvalues
                eig_idx = k_idx+2+n*(atoms+1)
                reig = re.findall('=\s+([+-]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)',file_list[eig_idx])[1]
                eig[k][n] = float(reig)
                for i in range(atoms):
                    #read eigenvectors
                    svec = re.findall('([+-]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)',file_list[eig_idx+1+i])
                    z = list(map(float,svec))
                    cvec = [complex(z[0],z[1]),complex(z[2],z[3]),complex(z[4],z[5])]
                    vec[k][n][i] = np.array(cvec, dtype=complex)

        self.nqpoints     = len(qpt)
        self.nphons       = nphons
        self.eigenvalues  = eig#*eV/hartree_cm1
        self.eigenvectors = vec.view(dtype=float).reshape([self.nqpoints,nphons,nphons,2])

        #convert to reduced coordinates
        self.qpoints = car_red(qpt,self.rec)
        return self.eigenvalues, self.eigenvectors, self.qpoints

    def read_atoms(self,filename):
        """ 
        read the data from a quantum espresso input file
        """
        pwin = PwIn(filename=filename[1:])
        cell, pos, self.atom_types = pwin.get_atoms()
        self.cell = np.array(cell)*bohr_angstroem
        self.rec = rec_lat(cell)*pwin.alat
        self.pos = np.array(pos)
        self.atom_numbers = [atomic_numbers[x] for x in self.atom_types]
        self.atomic_numbers = np.unique(self.atom_numbers)
        self.chemical_symbols = np.unique(self.atom_types).tolist()
        self.natoms = len(self.pos)
        self.chemical_formula = self.get_chemical_formula()

        pos_type = pwin.atomic_pos_type.lower()
        if pos_type == "cartesian":
            #convert to reduced coordinates
            self.pos = car_red(self.pos,self.cell)
        elif pos_type == "crystal" or pos_type == 'alat':
            #already in reduced coordinates
            pass
        else:
            raise ValueError("Coordinate format %s in input file not known"%pos_type)
#=========================================================
# End of phononutils.py
#=========================================================