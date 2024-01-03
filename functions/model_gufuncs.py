import sys
sys.path.append('.../')
from functions import model_device_functions as mdf

import numpy as np
from numba import guvectorize
import numba

@guvectorize(['float32[:], float32, float32, int32, int32, int32, int32, float64[:], float64[:]'], '(n), (), (), (), (), (), (), (k)->(k)', target='cuda') #the last argument is the output and therefore cannot be included in the input signature
def get_L_final_vec(parr4, h, Ti, Nb, n_max, num_steps, N_nucl, zero_vec, L_vec):

    found_L_vec = mdf.L(L_vec, zero_vec, parr4, N_nucl, Nb, n_max, num_steps, h, Ti)
    for n in range(n_max +2):
        L_vec[n] = found_L_vec[n]

    