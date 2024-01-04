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
    
def get_error_in_sig_fig(errors_vec):
    sigfigs = []
    for error in errors_vec:
        error_str = f'{error:.2e}' #round to 1 sig fig
        if error_str[0] == '1': #if the first sigfig is a 1
            sigfig = f'{error:.1e}' #round to 2 sig figs
        else:
            sigfig = f'{error:.0e}' #round to 1 sig fig
            if sigfig[0] == '1': #if the first sigfig is a 1 only after rounding up to one sigfig
                sigfig = sigfig.replace('1','1.0')
        sigfigs.append(sigfig)
    return sigfigs

def sign(num_in_scinotation):
    if num_in_scinotation[-3] == '-':
        return -1
    else:
        return 1
    
def get_d(error_str, mean_str): #get the difference in powers between 2 numbers
    extra_error_sig_figs = len(error_str.replace('.','')) - 5 #find the number of extra sig figs in the error
    d = int(mean_str[-1]) * sign(mean_str) - int(error_str[-1]) * sign(error_str) + extra_error_sig_figs
    return d

def get_mean_in_sig_fig(means_vec, errors_vec):
    sigfigs = []
    errors_str_vec = get_error_in_sig_fig(errors_vec)
    means_str_vec = [f'{mean:.2e}' for mean in means_vec]
    for i in range(len(means_vec)):
        if means_vec[i] > errors_vec[i]: #if the mean is larger than the error
            d = get_d(errors_str_vec[i], means_str_vec[i])        
            mean = means_vec[i]
            sigfig = f'{mean:.{d}e}'
        else: #if the error is larger than the mean
            sigfig = f'{means_vec[i]:.0e}'
        sigfigs.append(sigfig)
    return sigfigs

def plot_model_results(parrs, models, x_data, y_data_sets, y_error_sets, axs, colors, seed_len_types, parr_error = None):
    #chi_squares = [chi_square(y_data_sets[i], y_error_sets[i], models[i], conts, per_n=True) for i in range(len(seed_len_types))]
    
    for i in range(4)[::-1]: #for seed type
        y_axis = [0,0,1,1][i] #determine which quadrant the plot goes in
        x_axis = [0,1,0,1][i]

        axs[y_axis, x_axis].plot(x_data, models[i], color = colors[i], label=seed_len_types[i])
        #axs[y_axis, x_axis].set_title(f'$\chi^2 = {chi_squares[i]:.2f}$')
        
        col_labels = ['$k_p$', '$k_{{p0}}$', '$k_d$', '$k_{{break}}$']
        if parr_error == None:
            table_vals = [['%.2e' % j for j in parrs[i]]]
            table = axs[y_axis, x_axis].table(cellText = table_vals, bbox = [0.28,0.4,0.7,0.2], colLabels = col_labels, colWidths= [0.1]*4)
        else:
            table_vals = [get_mean_in_sig_fig(parrs[i], parr_error[i]),
                          [f'$\pm${error}' for error in get_error_in_sig_fig(parr_error[i])]]
                          
            #table_vals = [['%.2e' % j for j in parrs[i]],
            #             [ '$\pm$%.1g' % k for k in parr_error[i]]]
            table = axs[y_axis, x_axis].table(cellText = table_vals, bbox = [0.28,0.32,0.7,0.4], colLabels = col_labels, colWidths= [0.1]*4)

         
        table.auto_set_font_size(False)
        table.set_fontsize(18)
        
        axs[y_axis, x_axis].legend(prop={'size': 20}, frameon=False)