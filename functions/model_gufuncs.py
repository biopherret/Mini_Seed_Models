import sys
sys.path.append('.../')
from functions import model_device_functions as mdf

import numpy as np
from numba import guvectorize
import numba

@guvectorize(['float32[:], float32[:], int32[:], float64[:], float64[:]'], '(n), (m), (l), (k)->(k)', target='cuda') #exclude the last input object of the function in the signature otherwise will error
def get_L_final_vec(parr4, cont_floats, cont_ints, zero_vec, L_vec):

    Nb = cont_ints[0]
    n_max = cont_ints[1]
    num_steps = cont_ints[2]
    N_nucl = cont_ints[3]
    h = cont_floats[0]
    Ti = cont_floats[1]

    found_L_vec = mdf.L(L_vec, zero_vec, parr4, N_nucl, Nb, n_max, num_steps, h, Ti)
    for n in range(n_max +2):
        L_vec[n] = found_L_vec[n]

    