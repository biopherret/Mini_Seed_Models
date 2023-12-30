import sys
sys.path.append('.../')
from functions import model_device_functions as mdf

import numpy as np
from numba import guvectorize
import numba

@guvectorize(['float64[:,:], float32[:], int32, int32, int32, int32, float32, float32'], '(n,m), (l), (), (), (), (), ()->()', target='cuda')
def get_L_mat(L_mat, parr4, N_nucl, Nb, n_max, num_steps, h, Ti):
    #new_L_mat = numba.float32[:,:]
    #for t in range(num_steps):
    #    for n in range(n_max +2):
    #        new_L_mat[t,n] = 0 #establish the size of the matrix and fill with zeros
    found_L_mat = mdf.L(L_mat, parr4, N_nucl, Nb, n_max, num_steps, h, Ti)
    for t in range(num_steps):
        for n in range(n_max +2):
            L_mat[t,n] = found_L_mat[t,n]
    