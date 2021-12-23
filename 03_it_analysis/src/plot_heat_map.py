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

def plot_heat_map(matrix, mask_threshold, date_start, date_end, save_location, save=False):
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
            'label': 'Correlation'
           }
    heatmap = sns.heatmap(matrix_plot.transpose(), vmin = -1, vmax = 1, cbar = True, cmap='coolwarm', 
                      annot = True, cbar_kws = cbar_kws, linewidth = 1, mask = mask.transpose())
    plt.xticks(rotation=45,rotation_mode='anchor',ha = 'right')
    heatmap.set_title('|Correlation| > '+str(mask_threshold)+'\n '+date_start+' - '+date_end, fontdict={'fontsize':14}, pad=12)
    if save:
        plt.savefig(save_location, bbox_inches = 'tight')
    else:
        plt.show()
        
def main():
    # import config
    with open("03_it_analysis/plot_heat_map_config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)['plot_heat_map.py']
    
    matrix = config['matrix']
    mask_threshold = config['mask_threshold']
    date_start = config['date_start']
    date_end = config['date_end']
    save_location = config['save_location']
    save = config['save']
    
    plot_heat_map(matrix, mask_threshold, date_start, date_end, save_location, save)
    
if __name__ == '__main__':
    main()