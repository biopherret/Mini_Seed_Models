import numpy as np
from numba import cuda
import numba

@cuda.jit(device=True)
def vec_sum(vec):
    s = 0
    for i in range(len(vec)):
        s += vec[i]
    return s

@cuda.jit(device=True)
def scal_x_vec(scal, vec):
    for i in range(len(vec)):
        vec[i] *= scal
    return vec

@cuda.jit(device=True)
def vec_add(vec1, vec2):
    for i in range(len(vec1)):
        vec1[i] += vec2[i]
    return vec1

@cuda.jit(device=True)
def part_kp(L_vec, n, Nb, n_max):
    if n == n_max + 1:
        return -Nb * L_vec[n_max + 1] * vec_sum(L_vec[1:n_max])
    elif n==0:
        return 0
    elif n==1:
        return -L_vec[n_max + 1] * L_vec[1]
    else: #for 1<n<n_max +1
        return L_vec[n_max + 1] * (L_vec[n-1] - L_vec[n])
    
@cuda.jit(device=True)
def part_kp0(L_vec, n, N_nucl, Nb, n_max):
    if n==n_max + 1:
        return -(Nb - N_nucl) * L_vec[n_max + 1] * L_vec[0]
    elif n==0:
        return -L_vec[n_max + 1] * L_vec[0]
    elif n==1:
        return L_vec[n_max + 1] * L_vec[0]
    else: #1<n<n_max + 1
        return 0

@cuda.jit(device=True)
def part_kd(L_vec, n, N_nucl, Nb, n_max):
    if n==n_max + 1:
        return Nb * vec_sum(L_vec[2:n_max + 1]) + (Nb - N_nucl) * L_vec[1]
    elif n==0:
        return L_vec[1]
    elif n==1:
        return L_vec[2] - L_vec[1]
    elif n==n_max:
        return -L_vec[n_max]
    else: #1<n<n_max
        return L_vec[n+1] - L_vec[n]

@cuda.jit(device=True)
def part_kbreak(L_vec, n, n_max):
    if n==(n_max + 1) or n==0:
        return 0
    else:
        return 2 * vec_sum(L_vec[n+1:n_max + 1]) - (n-1) * L_vec[n]