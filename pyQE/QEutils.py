# -*- coding: utf-8 -*-
#=========================================================
# Beginning of QEutils.py
# @author: Tristan Britt
# @email: tristan.britt@mail.mcgill.ca
# @description: This file contains every util function 
# used by this package to read input from input QE SCF file
# and every other mathematical function used by this 
# package
#
#
# This software is part of a package distributed under the 
# GPLv3 license, see LICENSE.txt
#=========================================================
from math import isclose, sqrt
import numpy as np
import os
import re


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

    
e1, e2, e3 = np.eye(3) #standard basis
def change_of_basis(basis1, basis2=(e1, e2, e3)):
    """
    Returns the matrix transforms vectors expressed in one basis,
    to vectors expressed in another basis.

    Parameters
    ----------
    basis1 : list of array_like, shape (3,)
        First basis
    basis2 : list of array_like, shape (3,), optional
        Second basis. By default, this is the standard basis

    Returns
    -------
    cob : `~numpy.ndarray`, shape (3,3)
        Change-of-basis matrix.
    """
    # Calculate the transform that goes from basis 1 to standard basis
    basis1 = [np.asarray(vector).reshape(3, 1) for vector in basis1]
    basis1_to_standard = np.hstack(tuple(basis1))

    # Calculate the transform that goes from standard basis to basis2
    basis2 = [np.asarray(vector).reshape(3, 1) for vector in basis2]
    standard_to_basis2 = np.linalg.inv(np.hstack(tuple(basis2)))

    return np.dot(standard_to_basis2, basis1_to_standard)

def coth(*args, **kwargs):
    """Hyperbolic cotangent"""
    return np.cosh(*args, **kwargs) / np.sinh(*args, **kwargs)

def rowdot(arr, brr):
    """Row-wise dot product"""
    # This is much, much faster than np.inner for some reason.
    return np.einsum("ij,ij->i", arr, brr).reshape((-1, 1))


def mapply(matrix, table):
    """Apply a matrix transformation to a table where every row is considered a vector."""
    return np.transpose(matrix @ table.T)


def unique_by_row_idx(arr):
    """Return the indices of the unique rows in arr"""
    # np.unique can be rather slow when checking for uniqueness in rows
    # The optimization below results in ~3X faster performance
    #
    # A discussion of why is located here:
    #   https://github.com/numpy/numpy/issues/11136
    arr = np.ascontiguousarray(arr)
    arr_row_view = arr.view("|S%d" % (arr.itemsize * arr.shape[1]))
    _, unique_row_indices = np.unique(arr_row_view, return_index=True)
    return unique_row_indices


def unique_by_rows(*arrs):
    """Filter arrays by the unique rows of the first array"""
    unique_row_indices = unique_by_row_idx(arrs[0])
    return tuple(arr[unique_row_indices] for arr in arrs)



def roughly_unique_by_rows(*arrs, decimals, axis=0):
    """Apply uniqueness rules on an array, based on lower precision."""
    rough = np.copy(arrs[0])
    np.around(rough, decimals=decimals, out=rough)
    return unique_by_rows(rough, *arrs[1:])


def tile_over_rows(*arrs):
    """Tile arrays over rows time until all arrays have the same number of rows as the first array"""
    nrows = arrs[0].shape[0]

    arrays = [arrs[0]]
    for array in arrs[1:]:
        missing_reps = int(nrows / array.shape[0])
        reps_tuple = [1] * array.ndim
        reps_tuple[0] = missing_reps
        newarr = np.tile(array, reps=tuple(reps_tuple))
        arrays.append(newarr)

    return tuple(arrays)


def is_in_plane(transformation):
    """Determine if a symmetry transformation is in the a-b plane"""
    translation = transformation[0:3, -1]

    if not isclose(translation[2], 0, abs_tol=1e-5):
        return False

    return True


def remove_by_rows(remove, *arrs):
    """
    Filter arrays where rows in `remove` are True are removed.
    """
    for arr in arrs:
        assert remove.shape[0] == arr.shape[0]

    return tuple(arr[np.logical_not(remove)] for arr in arrs)

class hashabledict(dict):
    def __key(self):
        return tuple((k,self[k]) for k in sorted(self))
    def __hash__(self):
        return hash(self.__key())
    def __eq__(self, other):
        return self.__key() == other.__key()


class PwIn():
    """
    Class to generate an manipulate Quantum Espresso input files
    Can be initialized either reading from a file or starting from a new file.

    Examples of use:

    To read a local file with name "mos2.in"

        .. code-block :: python

            qe = PwIn('mos2.scf')
            print qe

    To start a file from scratch

        .. code-block :: python

            qe = PwIn('mos2.scf')
            qe.atoms = [['N',[ 0.0, 0.0,0.0]],
                        ['B',[1./3,2./3,0.0]]]
            qe.atypes = {'B': [10.811, "B.pbe-mt_fhi.UPF"],
                         'N': [14.0067,"N.pbe-mt_fhi.UPF"]}

            qe.control['prefix'] = "'%s'"%prefix
            qe.control['verbosity'] = "'high'"
            qe.control['wf_collect'] = '.true.'
            qe.control['pseudo_dir'] = "'../pseudos/'"
            qe.system['celldm(1)'] = 4.7
            qe.system['celldm(3)'] = layer_separation/qe.system['celldm(1)']
            qe.system['ecutwfc'] = 60
            qe.system['occupations'] = "'fixed'"
            qe.system['nat'] = 2
            qe.system['ntyp'] = 2
            qe.system['ibrav'] = 4
            qe.kpoints = [6, 6, 1]
            qe.electrons['conv_thr'] = 1e-8

            print qe

    Special care should be taken with string variables e.g. "'high'"

    """
    _pw = 'pw.x'

    def __init__(self, filename=None):
        #kpoints
        self.ktype = "automatic"
        self.kpoints = [1,1,1]
        self.shiftk = [0,0,0]
        self.klist = []
        #dictionaries
        self.control = dict()
        self.system = dict()
        self.electrons = dict()
        self.ions = dict()
        self.cell = dict()
        self.atypes = dict()
        self.atoms = []
        self.cell_parameters = []
        self.cell_units = 'angstrom'
        self.atomic_pos_type = 'crystal'

        #in case we start from a reference file
        if filename:
            f = open(filename,"r")
            self.file_name = filename #set filename
            self.file_lines = f.readlines() #set file lines
            self.store(self.control,"control")     #read &control
            self.store(self.system,"system")      #read &system
            self.store(self.electrons,"electrons")   #read &electrons
            self.store(self.ions,"ions")        #read &ions
            self.store(self.cell,"cell")        #read &ions
            #read ATOMIC_SPECIES
            self.read_atomicspecies()
            #read ATOMIC_POSITIONS
            self.read_atoms()
            #read K_POINTS
            self.read_kpoints()
            #read CELL_PARAMETERS
            self.read_cell_parameters()

    def read_atomicspecies(self):
        lines = iter(self.file_lines)
        #find ATOMIC_SPECIES keyword in file and read next line
        for line in lines:
            if line.replace(' ', '').startswith('ATOMIC SPECIES'):
                for i in range(int(self.system["ntyp"])):
                    atype, mass, psp = next(lines).split()
                    self.atypes[atype] = [mass,psp]

    def get_symmetry_spglib(self):
        """
        get the symmetry group of this system using spglib
        """
        import spglib

        lat, positions, atypes = self.get_atoms()
        lat = np.array(lat)

        at = np.unique(atypes)
        an = dict(list(zip(at,list(range(len(at))))))
        atypes = [an[a] for a in atypes]

        cell = (lat,positions,atypes)

        spacegroup = spglib.get_spacegroup(cell,symprec=1e-5)
        return spacegroup

    def get_masses(self):
        """ Get an array with the masses of all the atoms
        """
        masses = []
        for atom in self.atoms:
            atype = self.atypes[atom[0]]
            mass = float(atype[0])
            masses.append(mass)
        return masses

    def set_path(self,path):
        self.klist = path.get_klist()

    def get_atoms(self):
        """ Get the lattice parameters, postions of the atoms and chemical symbols
        """
        self.read_cell_parameters()
        cell = self.cell_parameters
        sym = [atom[0] for atom in self.atoms]
        pos = [atom[1] for atom in self.atoms]
        if self.atomic_pos_type == 'bohr':
            pos = car_red(pos,cell)
        return cell, pos, sym

    def set_atoms_string(self,string):
        """
        set the atomic postions using string of the form
        Si 0.0 0.0 0.0
        Si 0.5 0.5 0.5
        """
        atoms_str = [line.strip().split() for line in string.strip().split('\n')]
        self.atoms = []
        for atype,x,y,z in atoms_str:
            self.atoms.append([atype,list(map(float,[x,y,z]))])

    def set_atoms(self,atoms):
        """ set the atomic postions using a Atoms datastructure from ase
        """
        # we will write down the cell parameters explicitly
        self.system['ibrav'] = 0
        if 'celldm(1)' in self.system: del self.system['celldm(1)']
        self.cell_parameters = atoms.get_cell()
        self.atoms = list(zip(atoms.get_chemical_symbols(),atoms.get_scaled_positions()))
        self.system['nat'] = len(self.atoms)

    def displace(self,mode,displacement,masses=None):
        """ A routine to displace the atoms acoording to a phonon mode
        """
        if masses is None:
            masses = [1] * len(self.atoms)
            small_mass = 1
        else:
            small_mass = min(masses) #we scale all the displacements to the bigger mass
        for i in range(len(self.atoms)):
            self.atoms[i][1] = self.atoms[i][1] + mode[i].real*displacement*sqrt(small_mass)/sqrt(masses[i])

    def read_atoms(self):
        lines = iter(self.file_lines)
        #find READ_ATOMS keyword in file and read next lines
        for line in lines:
            if line.replace(' ', '').startswith('ATOMIC_POSITIONS'):
                atomic_pos_type = line
                self.atomic_pos_type = re.findall('([A-Za-z]+)',line)[-1]
                for i in range(int(self.system["nat"])):
                    atype, x,y,z = next(lines).split()
                    self.atoms.append([atype,[float(i) for i in (x,y,z)]])
        self.atomic_pos_type = atomic_pos_type.replace('{','').replace('}','').strip().split()[1]

    def read_cell_parameters(self):
        ibrav = int(self.system['ibrav'])
        def rmchar(string,symbols): return ''.join([c for c in string if c not in symbols])

        if ibrav == 0:
            if 'celldm(1)' in list(self.system.keys()):
                a = float(self.system['celldm(1)'])
            else:
                a = 1
            lines = iter(self.file_lines)
            for line in lines:
                if "CELL_PARAMETERS" in line:
                    units = rmchar(line.strip(),'{}()').split()
                    self.cell_parameters = [[],[],[]]
                    if len(units) > 1:
                        self.cell_units = units[1]
                    else:
                        self.cell_units = 'bohr'
                    for i in range(3):
                        self.cell_parameters[i] = [ float(x)*a for x in next(lines).split() ]
            if self.cell_units == 'angstrom' or self.cell_units == 'bohr':
                if 'celldm(1)' in self.system: del self.system['celldm(1)']
            
            if 'celldm(1)' not in list(self.system.keys()):
                a = np.linalg.norm(self.cell_parameters[0])
        elif ibrav == 1:
            a = float(self.system['celldm(1)'])
            self.cell_parameters = [[  a,   0,   0],
                                    [  0,   a,   0],
                                    [  0,   0,   a]]
        elif ibrav == 2:
            a = float(self.system['celldm(1)'])
            self.cell_parameters = [[ -a/2,   0, a/2],
                                    [    0, a/2, a/2],
                                    [ -a/2, a/2,   0]]
        elif ibrav == 3:
            a = float(self.system['celldm(1)'])
            self.cell_parameters = [[ a/2,  a/2,  a/2],
                                    [-a/2,  a/2,  a/2],
                                    [-a/2, -a/2,  a/2]]
        elif ibrav == 4:
            a = float(self.system['celldm(1)'])
            c = float(self.system['celldm(3)'])
            self.cell_parameters = [[   a,          0,  0],
                                    [-a/2,sqrt(3)/2*a,  0],
                                    [   0,          0,c*a]]
        elif ibrav == 6:
            a = float(self.system['celldm(1)'])
            c = float(self.system['celldm(3)'])
            self.cell_parameters = [[  a,   0,   0],
                                    [  0,   a,   0],
                                    [  0,   0, c*a]]
        elif ibrav==8:
            try:
                a = float(self.system['celldm(1)'])
                b = float(self.system['celldm(2)'])
                c = float(self.system['celldm(3)'])
                self.cell_parameters = [[a,   0,   0],
                                        [0, a*b,   0],
                                        [0,   0, a*c]]
            except KeyError:
                a = float(self.system['a'])
                b = float(self.system['b'])
                c = float(self.system['c'])
                self.cell_parameters = [[a,   0,   0],
                                        [0,   b,   0],
                                        [0,   0,   c]]
        else:
            raise NotImplementedError('ibrav = %d not implemented'%ibrav)
        self.alat = a 
        
    def read_kpoints(self):
        lines = iter(self.file_lines)
        #find K_POINTS keyword in file and read next line
        for line in lines:
            if "K_POINTS" in line:
                #chack if the type is automatic
                if "automatic" in line:
                    self.ktype = "automatic"
                    vals = list(map(float, next(lines).split()))
                    self.kpoints, self.shiftk = vals[0:3], vals[3:6]
                #otherwise read a list
                else:
                    #read number of kpoints
                    nkpoints = int(next(lines).split()[0])
                    self.klist = []
                    self.ktype = ""
                    try:
                        lines_list = list(lines)
                        for n in range(nkpoints):
                            vals = lines_list[n].split()[:4]
                            self.klist.append( list(map(float,vals)) )
                    except IndexError:
                        print("wrong k-points list format")
                        exit()

    def slicefile(self, keyword):
        return re.findall('&(?i)%s(?:.?)+\n((?:.+\n)+?)(?:\s+)?[\/&]'%keyword,"".join(self.file_lines),re.MULTILINE)

    def store(self,group,name):
        """
        Save the variables specified in each of the groups on the structure
        """
        for file_slice in self.slicefile(name):
            for keyword, value in re.findall('([a-zA-Z_0-9_\(\)]+)(?:\s+)?=(?:\s+)?([a-zA-Z/\'"0-9_.-]+)',file_slice):
                group[keyword.strip()]=value.strip()

    def stringify_group(self, keyword, group):
        if group != {}:
            string='&%s\n' % keyword
            for keyword in sorted(group): # Py2/3 discrepancy in keyword order
                string += "%20s = %s\n" % (keyword, group[keyword])
            string += "/&end\n"
            return string
        else:
            return ''

    def remove_key(self,group,key):
        """ if a certain key exists in the group, remove it
        """
        if key in list(group.items()):
            del group[key]

    def run(self,filename,procs=1,folder='.'):
        """ this function is used to run this job locally
        """
        os.system('mkdir -p %s'%folder)
        self.write("%s/%s"%(folder,filename))
        if procs == 1:
            os.system('cd %s; OMP_NUM_THREADS=1 %s -inp %s > %s.log' % (folder,self._pw,filename,filename))
        else:
            os.system('cd %s; OMP_NUM_THREADS=1 mpirun -np %d %s -inp %s > %s.log' % (folder,procs,self._pw,filename,filename))

    def write(self,filename):
        f = open(filename,'w')
        f.write(str(self))
        f.close()

    def __str__(self):
        """
        Output the file in the form of a string
        """
        string = ''
        string += self.stringify_group("control",self.control) #print control
        string += self.stringify_group("system",self.system) #print system
        string += self.stringify_group("electrons",self.electrons) #print electrons
        string += self.stringify_group("ions",self.ions) #print ions
        string += self.stringify_group("cell",self.cell) #print ions

        #print atomic species
        string += "ATOMIC_SPECIES\n"
        for atype in self.atypes:
            string += " %3s %8s %20s\n" % (atype, self.atypes[atype][0], self.atypes[atype][1])
        #print atomic positions
        string += "ATOMIC_POSITIONS { %s }\n"%self.atomic_pos_type
        for atom in self.atoms:
            string += "%3s %14.10lf %14.10lf %14.10lf\n" % (atom[0], atom[1][0], atom[1][1], atom[1][2])
        #print kpoints
        if self.ktype == "automatic":
            string += "K_POINTS { %s }\n" % self.ktype
            string += ("%3d"*6+"\n")%tuple(self.kpoints + self.shiftk)
        elif self.ktype == "crystal":
            string += "K_POINTS { %s }\n" % self.ktype
            string += "%d\n" % len(self.klist)
            for i in self.klist:
              string += ('%12.8lf '*4+'\n') % tuple(i)
        else:
            string += "K_POINTS { }\n"
            string += "%d\n" % len(self.klist)
            for i in self.klist:
                string += (("%12.8lf "*4)+"\n")%tuple(i)
        if self.system['ibrav'] == 0 or self.system['ibrav'] == '0':
            string += "CELL_PARAMETERS %s\n"%self.cell_units
            for i in range(3):
                string += ("%14.10lf "*3+"\n")%tuple(self.cell_parameters[i])
        return string
#=========================================================
# End of QEutils.py
#=========================================================