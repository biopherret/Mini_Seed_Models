import json
import numpy as np
from scipy import integrate
import sys
from functions import model_gufuncs as mgpu

def write_json(data, file_name):
    with open (file_name, 'w') as file:
        json.dump(data, file, indent = 4)

def open_json(file_name):
    with open (file_name) as file:
        return json.load(file)
    
def find_bin_count_error(len_dist, x_data, mes_error):
    '''Find the error in the bin count for a given (not normalized) length distribution'''
    bin_count_error = []
    bin_half_width = (x_data[1]-x_data[0])/2
    for bin in x_data:
        bin_stack = []
        for len in len_dist:
            if len >= bin - bin_half_width and len <= bin + bin_half_width:
                bin_stack.append(len)
        s = 0
        for len in bin_stack:
            integrad = lambda z : (np.exp((-(len - z)**2)/(2 * mes_error**2)))/(np.sqrt(2 * np.pi) * mes_error)
            integral = integrate.quad(integrad, bin - bin_half_width, bin + bin_half_width)[0]
            s += integral * (1 - integral)
        bin_count_error.append(s)
    return bin_count_error

def get_os_length_dicts(seed_len_types):
    '''Get the length distributions for one sided seeds from the data dict json file'''

    data = open_json('data_dict.json')

    len_dists = []
    for i in range(len(seed_len_types)): #for seed type
        len_dist = []
        seed_len = seed_len_types[i]
        for slide_sample_id in data[f'os {seed_len}']: #for each sample
            sample = data[f'os {seed_len}'][slide_sample_id]
            len_dist.extend(sample['length_distribution']) #combine the length distributions into one list
    
        len_dists.append(len_dist)
    return len_dists

def get_ydata(len_dist, bins, x_data):
    y_data, _= np.histogram(len_dist, bins = bins) #grab the resulting y points by plotting the histogram
    normed_y_data = y_data/find_area_under_curve(x_data, y_data) #normalize the data
    return normed_y_data

def get_yerror(len_dist, bins, x_data, mes_error):
    y_data, _= np.histogram(len_dist, bins = bins) #grab the resulting y points by plotting the histogram
    y_error = find_bin_count_error(len_dist, x_data, mes_error)
    
    normed_y_error = normalize_hist_error(x_data, y_data, y_error)
    return normed_y_error

def normalize_hist_error(x, y, y_err):
    bin_width = x[1]-x[0] #bin width is constant
    delta_A = bin_width * np.sqrt(np.sum([Ni**2 for Ni in y_err])) #calculate the error of the area under the curve
    A = find_area_under_curve(x, y) #calculate the area under the curve
    return (y * delta_A)/(A**2) #normalize the error

def find_area_under_curve(x, y):
    bin_width = x[1]-x[0] #bin width is constant
    A = bin_width * np.sum(y) #calculate the area under the curve
    return A

def chi_square(y_data, y_error, model, n_max, per_n = False):
    s = 0
    n_num = 0
    for n in range(n_max):
        if y_error[n] != 0:
            n_num += 1
            s += ((y_data[n] - model[n])/(y_error[n]))**2
    
    if per_n:
        return s/n_num
    else:
        return s

def log_like(L_mat, x_data, y_data_set, y_error_set, n_max):
    L_vec = L_mat[0]
    model = L_vec[1:n_max + 1] / find_area_under_curve(x_data, L_vec[1:n_max + 1])
    chi = chi_square(y_data_set, y_error_set, model, n_max, per_n = False)
    return -0.5 * (chi)**2