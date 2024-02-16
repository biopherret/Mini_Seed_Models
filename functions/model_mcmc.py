import matplotlib.pyplot as plt
import numpy as np
from functions import model_general as mg
from functions import model_gufuncs as mgpu

def take_random_steps(current_positions, sigma_guess):
    nwalkers = len(current_positions)
    ndim = len(sigma_guess)
    random_steps = np.zeros((nwalkers, ndim))
    for n in range(nwalkers): #for each walker
        random_param = np.random.randint(0,ndim) #choose a random parameter to change
        random_step = np.zeros(ndim)
        random_step[random_param] = sigma_guess[random_param] * np.random.normal() #take a step in the chosen direction
        random_steps[n] = random_step
        
    return random_steps

def plot_chains(chains):
    nwalkers = len(chains)
    fig, axs = plt.subplots(2,2, figsize = (25,10))
    for walker in range(nwalkers):
        chain = np.array(chains[walker])
        axs[0,0].plot(chain[:,0], color = 'r', alpha = 0.3) #kp
        axs[0,1].plot(chain[:,1], color = 'b', alpha = 0.3) #kp0
        axs[1,0].plot(chain[:,2], color = 'violet', alpha = 0.3) #kd
        axs[1,1].plot(chain[:,3], color = 'orange', alpha = 0.3) #kb
    axs[0,0].set_title('kp')
    axs[0,1].set_title('kp0')
    axs[1,0].set_title('kd')
    axs[1,1].set_title('kbreak')
    plt.show()

def run_mcmc(n_mcmc, nwalkers, ndim, chain_start, sigma_guess, N_nucl, L_mat0, Nb, n_max, num_steps, h, Ti, Si,x_data, y_data_set, y_error_set):
    chains = [[chain_start] for _ in range(nwalkers)]

    #get the starting positions and likelihoods
    positions = [chain[-1] for chain in chains] #get the current position
    position_lls = [mg.log_like(L_mat0, x_data, y_data_set, y_error_set, n_max) for _ in range(nwalkers)]

    for _ in range(n_mcmc):
        dx = take_random_steps(positions, sigma_guess)
 
        new_params = positions + dx

        new_L_mats = mgpu.get_L_mat(new_params, N_nucl, Nb, n_max, num_steps, h, Ti, Si)
        new_lls = np.array([mg.log_like(L_mat, x_data, y_data_set, y_error_set, n_max) for L_mat in new_L_mats])
    
        for walk in range(nwalkers):
            p_to_accept = np.exp(new_lls[walk] - position_lls[walk])
            if np.random.rand() < p_to_accept:
                positions[walk] = new_params[walk]
                position_lls[walk] = new_lls[walk]
                chains[walk].append(positions[walk])
        
    return chains

def flatten_chains(chains):
    nwalkers = len(chains)
    flat_chains = []
    for i in range(nwalkers):
        for step in chains[i]:
            flat_chains.append(step)
    return np.array(flat_chains)