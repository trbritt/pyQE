#This example creates a json just exclusively of the values required to compute one-phonon structure factors

def test_JSON(fname):

    from pyQE.phononutils import compute_onephonon_JSON
    params = {
        'Filename' : '/home//trbritt//Desktop//MoS2//HT-MoS2//GGA//03_phononic_properties//matdyn//',
        'IFC'  : '../../q2r/HT-MoS2_PAW444.fc', #relative to directory above down one more
        'matdyn_exec' : '~//q-e-qe-6.5//bin//matdyn.x', #matdyn executable
        'NSCF_out' : '/home//trbritt//Desktop//MoS2//preliminary_data//one_phonon//mos2.out',
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

    complete_freq, complete_qpoint, complete_polar_real, complete_polar_imaginary = compute_onephonon_JSON(params, fname)

if __name__ == "__main__":
    test_JSON('complete_Data_format1.json')