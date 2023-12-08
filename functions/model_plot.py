import numpy as np

def plot_hist(x_data, y_data_sets, y_error_sets, axs, seed_len_types, boot=False, norm = True):
    for i in range(len(seed_len_types)): #for each seed type
        y_axis = [0,0,1,1][i] #determine which quadrant the plot goes in
        x_axis = [0,1,0,1][i]

        y_data = y_data_sets[i] #get the seed's y data
        y_error = y_error_sets[i] #get the seed's error bars

        axs[y_axis, x_axis].bar(x_data, y_data, alpha = 0.3, label = 'SEs', color = 'g', yerr = y_error, width = 0.5, capsize = 5)
        axs[y_axis, x_axis].set_title(seed_len_types[i])
        axs[y_axis, x_axis].set_xlim(1,15)
        if boot:
            axs[y_axis, x_axis].set_ylim(0,0.9)
        if norm:
            axs[y_axis, x_axis].set_ylim(0,1.2)
            axs[1,0].set_ylabel('Normalized Frequency', fontsize = 20)
        else: 
            axs[1,0].set_ylabel('Frequency', fontsize = 20)

        axs[y_axis, x_axis].set_xticks(np.arange(1, 16, step = 1), np.arange(1, 16, step = 1))
        axs[y_axis, x_axis].legend(prop={'size': 25}, frameon=False)

    axs[1,0].set_xlabel('Length ($\mu m$)', fontsize = 20)