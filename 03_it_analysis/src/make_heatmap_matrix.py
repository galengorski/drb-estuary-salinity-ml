# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 15:40:35 2021

@author: ggorski
"""
import numpy as np
import pandas as pd
import pickle
import TEpython_ParallelNAN2
import yaml


###
def create_correlation_matrix(srcs_lagged_pickle, snks_list_pickle):
    '''Takes in the lagged sources and calculates the correlation between them and the
    sinks. Pearson correlation is used'''
    with open(srcs_lagged_pickle, 'rb') as src:
        srcs_list_lagged = pickle.load(src)
    with open(snks_list_pickle, 'rb') as snk:
        snks_list = pickle.load(snk)
    
    #select data preprocessed for correlation
    srcs_list_lagged_corr = srcs_list_lagged['correlation']
    snks_list_corr = snks_list['correlation']    
    
    corrs = pd.DataFrame()
    for sc in range(len(srcs_list_lagged_corr)):
        col_snks = pd.DataFrame()
        for sk in range(len(snks_list_corr)):
            sc_sk_corr = srcs_list_lagged_corr[sc].apply(lambda s: snks_list_corr[sk].corrwith(s))
            col_snks = col_snks.append(sc_sk_corr)
        corrs[list(col_snks.columns)] = col_snks
    return corrs 

###
def create_mutual_information_matrix(srcs_lagged_pickle, snks_list_pickle):
    '''Takes in the lagged sources and calculates the mutual information between them and the
    sinks. mutual information is used'''
    bins  = [11,11,11]
    with open(srcs_lagged_pickle, 'rb') as src:
        srcs_list_lagged = pickle.load(src)
    with open(snks_list_pickle, 'rb') as snk:
        snks_list = pickle.load(snk)
        
    #select data preprocessed for mutual information
    srcs_list_lagged_mi = srcs_list_lagged['mutual_information']
    snks_list_mi = snks_list['mutual_information']  

    a_list = []
    
    for sc in range(len(srcs_list_lagged_mi)):
        a = srcs_list_lagged_mi[sc]
        
        dfs = pd.DataFrame()
        for sk in range(len(snks_list_mi)):
            
            b = snks_list_mi[sk]
            MI_array = np.zeros((a.shape[1], b.shape[1]))
            
            
            for i,src_name in enumerate(a):
                
                for j,snk_name in enumerate(b):
                    temp_src = a[src_name]
                    temp_snk = b[snk_name]
                    paired = temp_src.to_frame().join(temp_snk).to_numpy()
                    MI, n = TEpython_ParallelNAN2.mutinfo_new(paired, nbins = bins)
                    MI_array[i,j] = MI
                    #print(str(i)+' | '+str(j)+' | '+src_name+' | '+snk_name+'| MI = '+str(MI))
            mat = pd.DataFrame(MI_array.T, columns = a.columns)
            mat = mat.set_index(b.columns)
    
            dfs = dfs.append(mat)
        a_list.append(dfs)
    
    mi_mat = pd.concat(a_list, axis = 1)     
    
    return mi_mat

###
def main():
    # import config
    with open("03_it_analysis/make_heatmap_matrix_config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)['make_heatmap_matrix.py']
    
    #select the sources
    srcs_lagged_pickle = config['srcs_lagged_pickle']
    #select the sinks
    snks_list_pickle = config['snks_list_pickle']
    #select the out_directory
    out_dir = config['out_dir']
    
    #create correlation matrix
    corr_matrix = create_correlation_matrix(srcs_lagged_pickle, snks_list_pickle)
    #create mutual information matrix
    mi_matrix = create_mutual_information_matrix(srcs_lagged_pickle, snks_list_pickle)
    
    #save correlation matrix
    corr_matrix_save = open(out_dir+'corr_matrix', "wb")
    pickle.dump(corr_matrix, corr_matrix_save)
    corr_matrix_save.close()
    
    #save mutual information matrix
    mi_matrix_save = open(out_dir+'mi_matrix', "wb")
    pickle.dump(mi_matrix, mi_matrix_save)
    mi_matrix_save.close()
        
        
if __name__ == '__main__':
    main()
