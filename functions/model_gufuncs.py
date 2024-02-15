import sys
sys.path.append('.../')
from functions import model_device_functions as mdf

from numba import guvectorize, float64, int32, float32
import numpy as np

@guvectorize(
    [(float64[:], int32, int32, int32, int32, float32, float64[:,:], float64[:,:])],
    "(n), (), (), (), (), (), (m,l) -> (m,l)", target="cuda")
def _get_L_mat(parr, N_nucl, Nb, n_max, num_steps, h, L0_mat, L_mat):
    for n in range(n_max + 2):
        for l in range(2):
            L_mat[l,n] = L0_mat[l,n]

    for t_step in range(1,num_steps):
        for n in range(n_max + 2):
            pkp = mdf.part_kp(L_mat[0], n, Nb, n_max)
            pkp0 = mdf.part_kp0(L_mat[0], n, N_nucl, Nb, n_max)
            pkd = mdf.part_kd(L_mat[0], n, N_nucl, Nb, n_max)
            pkb = mdf.part_kbreak(L_mat[0], n, n_max)
            L_mat[1,n] = parr[0] * pkp + parr[1] * pkp0 + parr[2] * pkd + parr[3] * pkb
        for n in range(n_max + 2):
            L_mat[0,n] = mdf.vec_add(L_mat[0], mdf.scal_x_vec(h, L_mat[1]))[n]

def get_L_mat(parr, N_nucl, Nb, n_max, num_steps, h, Ti, Si):
    L0_vec = np.zeros((n_max+2), dtype=np.float64)
    L0_vec[n_max + 1] = Ti
    L0_vec[0] = Si

    dL0_vec = np.zeros((n_max+2), dtype=np.float64)

    L0_mat = np.array([L0_vec, dL0_vec], dtype=np.float64)
    return _get_L_mat(parr, N_nucl, Nb, n_max, num_steps, h, L0_mat)