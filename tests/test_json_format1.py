#This example creates a json just exclusively of the values required to compute one-phonon structure factors
from pyQE.phononutils import computeJSON
import json
params = {
    'Filename' : '/home//trbritt//Desktop//MoS2//HT-MoS2//GGA//03_phononic_properties//matdyn//',
    'nmb_atoms' : 3,
    'File' : 'MoS2_GGA',
    'Path' : ['M', 'K'],
    'Gamma' : True,
    'Start q' : 'X',
    'points' : 30000,
    'Start' : True,
    'Factor' : 600,
    'Get_ends' : True,
}

complete_freq, complete_qpoint, complete_polar_real, complete_polar_imaginary = computeJSON(params)
new_list = {'frequencies' : complete_freq.tolist(), 'kpoints' : complete_qpoint.tolist(), 'polarizations real' : complete_polar_real, 'polarizations imaginary' : complete_polar_imaginary}
with open(params['Filename'] + '//complete_Data_format1.json', 'w') as fil:
    json.dump(new_list, fil)

