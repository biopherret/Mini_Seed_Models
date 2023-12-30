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

@cuda.jit(device=True)
def part_dL(L_vec, n, N_nucl, par_div, Nb, n_max):
    if par_div == 'kp':
        return part_kp(L_vec, n, Nb, n_max)
    elif par_div == 'kp0':
        return part_kp0(L_vec, n, N_nucl, Nb, n_max)
    elif par_div == 'kd':
        return part_kd(L_vec, n, N_nucl, Nb, n_max)
    elif par_div == 'kbreak':
        return part_kbreak(L_vec, n, n_max)

@cuda.jit(['float64[:](float64[:], float64[:], float32[:], int32, int32, int32)'], device=True)
def dL(dL_vec, L_vec, parr, N_nucl, Nb, n_max):
    for n in range(n_max + 2):
        dL_vec[n] = parr[0] * part_dL(L_vec, n, N_nucl, 'kp', Nb, n_max) + parr[1] * part_dL(L_vec, n, N_nucl, 'kp0', Nb, n_max) + parr[2] * part_dL(L_vec, n, N_nucl, 'kd', Nb, n_max) + parr[3] * part_dL(L_vec, n, N_nucl, 'kbreak', Nb, n_max)
    
    return dL_vec

@cuda.jit(['float64[:,:](float64[:,:], float32[:], int32, int32, int32, int32, float32, float32)'], device=True)
def L(L_mat, parr, N_nucl, Nb, n_max, num_steps, h, Ti): 
    L_mat[0,n_max + 1] = 100 * 10**(-9) #initial [T] 100nM
    L_mat[0,0] = Ti

    L_vec = numba.float64[:]
    for n in range(n_max + 2):
        L_vec[n] = 0

    L_vec[n_max + 1] = 100 * 10**(-9)
    L_vec[0] = Ti
    for t_step in range(1,num_steps):
        new_dL_vec = numba.float64[:] #create an empty dL vector
        for n in range(n_max + 2):
            new_dL_vec[n] = 0
        
        L_vec = vec_add(L_vec, scal_x_vec(h, dL(new_dL_vec, L_vec, parr, N_nucl, Nb, n_max)))  #L_vec is from the prvious loop so this is calling dL at t_step-1
        for n in range(n_max + 2):
            L_mat[t_step,n] = L_vec[n]   
    return L_mat