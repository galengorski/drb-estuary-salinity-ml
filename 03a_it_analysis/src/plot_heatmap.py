# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 16:20:06 2021

@author: ggorski
"""
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns
import yaml

def plot_heatmap(matrix, mask_threshold, metric, date_start, date_end, save_location, vmin_val, vmax_val, save=False):
    '''Takes in a correlation matrix and plots a heat map of correlations,
    mask_threshold is a threshold where if abs(correlation) < threshold those
    squares of the heatmap are masked out to highlight stronger correlation. save is
    true/false whether it will save the plot to file, default false'''

    with open(matrix, 'rb') as mat:
        matrix_plot = pickle.load(mat)
        
    mask = abs(matrix_plot) < mask_threshold
    plt.figure(figsize = (5,10))
    cbar_kws = {"shrink":0.85,
            'extendfrac':0.1,
            'label': metric
           }
    heatmap = sns.heatmap(matrix_plot.transpose(), vmin = vmin_val, vmax = vmax_val, cbar = True, cmap='coolwarm', 
                      annot = True, cbar_kws = cbar_kws, linewidth = 1, mask = mask.transpose())
    plt.xticks(rotation=45,rotation_mode='anchor',ha = 'right')
    plt.yticks(rotation = 'horizontal')
    heatmap.set_title('|'+metric+'| > '+str(mask_threshold)+'\n '+date_start+' - '+date_end, fontdict={'fontsize':14}, pad=12)
    if save:
        plt.savefig(save_location, bbox_inches = 'tight')
    else:
        plt.show()
        
def main():
    # import config from data prep step
    with open("03a_it_analysis/it_analysis_data_prep_config.yaml", 'r') as stream:
        config_data_prep = yaml.safe_load(stream)['it_analysis_data_prep.py']
    
    #import config file 
    with open("03a_it_analysis/plot_heatmap_config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)['plot_heatmap.py']
    
    date_start = config_data_prep['date_start']
    date_end = config_data_prep['date_end']
    save = config['save']
    
    #make and save mutual information heatmap
    save_location = config['save_location']+'mutual_information.png'
    mi_matrix = config['mi_matrix']
    mask_threshold_mi = config['mask_threshold_mi']
    plot_heatmap(mi_matrix, mask_threshold_mi, 'mutual_information', date_start, date_end, save_location, 0,1,save)

    #make and save correlation heat map    
    save_location = config['save_location']+'correlation.png'
    corr_matrix = config['corr_matrix']
    mask_threshold_corr = config['mask_threshold_corr']
    plot_heatmap(corr_matrix, mask_threshold_corr, 'correlation', date_start, date_end, save_location, -1,1,save)
    
    
    
if __name__ == '__main__':
    main()