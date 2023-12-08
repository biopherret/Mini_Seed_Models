import numpy as np
from numba import cuda

#@cuda.jit(device=True)
def find_area_under_curve(x, y):
    bin_width = x[1]-x[0] #bin width is constant
    A = bin_width * np.sum(y) #calculate the area under the curve
    return A
