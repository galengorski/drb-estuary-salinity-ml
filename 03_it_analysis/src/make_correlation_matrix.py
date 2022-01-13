# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 15:40:35 2021

@author: ggorski
"""
import pandas as pd
import pickle
import yaml



def create_correlation_matrix(srcs_lagged_pickle, snks_list_pickle):
    '''Takes in the lagged sources and calculates the correlation between them and the
    sinks. Pearson correlation is used'''
    with open(srcs_lagged_pickle, 'rb') as src:
        srcs_list_lagged = pickle.load(src)
    with open(snks_list_pickle, 'rb') as snk:
        snks_list = pickle.load(snk)
    corrs = pd.DataFrame()
    for sc in range(len(srcs_list_lagged)):
        col_snks = pd.DataFrame()
        for sk in range(len(snks_list)):
            sc_sk_corr = srcs_list_lagged[sc].apply(lambda s: snks_list[sk].corrwith(s))
            col_snks = col_snks.append(sc_sk_corr)
        corrs[list(col_snks.columns)] = col_snks
    return corrs 

def main():
    # import config
    with open("03_it_analysis/make_correlation_matrix_config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)['make_correlation_matrix.py']
        
    #select the sources
    srcs_lagged_pickle = config['srcs_lagged_pickle']
    #select the sinks
    snks_list_pickle = config['snks_list_pickle']
    #select the out_directory
    out_dir = config['out_dir']
    corr_matrix = create_correlation_matrix(srcs_lagged_pickle, snks_list_pickle)
    
    corr_matrix_save = open(out_dir+'corr_matrix', "wb")
    pickle.dump(corr_matrix, corr_matrix_save)
    corr_matrix_save.close()
        
        
if __name__ == '__main__':
    main()
