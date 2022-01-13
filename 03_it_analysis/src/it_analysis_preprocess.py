# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 12:32:26 2022

@author: ggorski
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import yaml
###
class pre_proc_func:
    def __init__(self):
        pass

    def standardize(self, sr):
        standardized = list()
        for i in range(len(sr)):
            #standardize
            value_z = (sr[i] - np.nanmean(sr))/np.nanstd(sr, ddof = 1)
            standardized.append(value_z)
        return standardized

    def normalize(self, sr):
        normalized = list()
        for i in range(len(sr)):
            #normalize
            value_z = (sr[i]-np.nanmin(sr))/(np.nanmax(sr)-np.nanmin(sr))
            normalized.append(value_z)
        return normalized

    def remove_seasonal_signal(raw_data,j):
        seasonal_removed = list()
        for i in range(len(raw_data)):
            doy = raw_data.loc[i,'julianday']
            doy_mean = np.nanmean(raw_data[raw_data['julianday'] == doy].iloc[:,j])
            value = raw_data.iloc[i,j]-doy_mean
            seasonal_removed.append(value)
            seasonal_removed_np = np.array(seasonal_removed)
        return seasonal_removed_np

###
def apply_preprocessing_functions(var_list, source_sink, out_dir):
    '''Apply preprocessing functions to each variable. The preprocessing functions
    and the order that they are applied in are stored in the it_analysis_preprocess_config.yaml
    file. For each step in the process it saves a histogram of that data to the out_dir.
    The available preprocessing functions are of the class pre_proc_func and can be custom
    designed within the it_analysis_preprocess.py file. This function (apply_preprocessing_functions) has
    inputs and outputs that are structurally identical. Input is a list of dataframes of 'raw' data
    and output is a list of dataframes of processed data'''

    #assert (source_sink == 'sources'|source_sink == 'sinks'),'variable source_sink must be set to "sources" or "sinks"'
    #import config
    with open("03_it_analysis/it_analysis_preprocess_config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)['it_analysis_preprocess.py']
        if source_sink == 'sources':
            pre_process_steps = config['sources']
        else:
            pre_process_steps = config['sinks']

        #get the functions of class pre_proc_func
        ppf = globals()['pre_proc_func']()
        
        #save_dir = "03_it_analysis/out/"
    #make an ouptut directory for the histograms
    os.makedirs(out_dir+'preproces_plots', exist_ok=True)
    
    #make new empty list of processed source data
    data_proc_list = []
    #for each entry in the list, likely each site
    for var_num, var_data in enumerate(var_list):
        print(var_num)
        #create a new data frame with the same structure as the raw data
        var_proc_df = pd.DataFrame().reindex_like(var_list[var_num])
        #for each variable in that dataframe
        for var in list(var_proc_df.columns):
            #select that variable from the config file
            #if it says do nothing then
            if pre_process_steps[var][0] == 'none':
                #create histogram of raw data
                plt.hist(var_list[var_num][var])
                plt.ylabel('Count')
                plt.title(var + ' Raw')
                plt.savefig(out_dir+'preproces_plots/' +var + '_Raw.png', bbox_inches = 'tight')
                #plt.show()
                plt.close()
                
                #store the variable's data
                raw_data = var_list[var_num][var].copy()
                var_proc_df[var] = raw_data
                print('nothing to do here')
            else:
                #if there are preprocessing steps to do
                #store the data in temp data
                temp_data = var_list[var_num][var].copy()
                
                #create histogram of raw data
                plt.hist(temp_data)
                plt.ylabel('Count')
                plt.title(var + ' Raw')
                plt.savefig(out_dir+'preproces_plots/' +var + '_0_Raw.png', bbox_inches = 'tight')
                #plt.show()
                plt.close()
                
                #for each preprocessing step for that variable
                for count,value in enumerate(pre_process_steps[var]):
                    #apply the function
                    func = getattr(ppf,value)
                    temp_data = func(temp_data)
                    
                    #create histogram
                    plt.hist(temp_data)
                    plt.ylabel('Count')
                    plt.title(var + ' ' + value + ' step ' + str(count+1)+ '/'+ str(len(pre_process_steps[var])))
                    plt.savefig(out_dir+'preproces_plots/' +var + '_'+str(count+1)+' '+ value+'.png', bbox_inches = 'tight')
                    #plt.show()
                    plt.close()
                    
                    #store the variable's data
                    var_proc_df[var] = temp_data
                    print("apply: "+value)
        
        #add the processed data into the processed list
        data_proc_list.append(var_proc_df)
    return(data_proc_list)
###
def save_proc_sources_sinks(srcs_proc_list, snks_proc_list, out_dir):
    #create out directory if it doesn't already exist
    os.makedirs(out_dir, exist_ok = True)
    #write snks_list to file 
    snks_file = open(out_dir+'snks_proc', "wb")
    pickle.dump(snks_proc_list, snks_file)
    snks_file.close()
    #write srcs_list_lagged to file 
    srcs_file = open(out_dir+'srcs_proc_lagged', "wb")
    pickle.dump(srcs_proc_list, srcs_file)
    srcs_file.close()
    print('sinks and sources saved to file')


###
def lag_sources(n_lags, srcs_list):
    '''Takes in the list of sources called srcs_list in which each list item is a 
    dataframe of site variables, and creates lagged time series for all variables
    according to the n_lag variable. Column headers are updted with the n_lag. n_lag refers
    to the number of time steps to lag, and it is agnostic of the time resolution of the data.
    length of data > n_lag >=0'''
    #lag the source variables
    for s in range(len(srcs_list)):
        for col_name in list(srcs_list[s].columns):
            for lag in range(1, n_lags+1):
                #create the lagged time series and name the columns
                srcs_list[s][col_name+'_lag_'+str(lag)] = srcs_list[s][col_name].shift(lag)
        #sort the varaibles for ease of plotting
        srcs_list[s] = srcs_list[s].sort_index(axis=1)
    return srcs_list

###
def main():
    #import config
    with open("03_it_analysis/it_analysis_preprocess_config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)['it_analysis_preprocess.py']
    #read in sources list
    with open('03_it_analysis/out/srcs', 'rb') as src:
        srcs_list = pickle.load(src)
    #read in sinks list
    with open('03_it_analysis/out/snks', 'rb') as snk:
        snks_list = pickle.load(snk)
    out_dir = config['out_dir'] 
    #process the sources
    srcs_proc = apply_preprocessing_functions(srcs_list, 'sources', out_dir)
    #process the sinks
    snks_proc = apply_preprocessing_functions(snks_list, 'sinks', out_dir)
    #number of lag days to consider for sources
    n_lags = config['n_lags']
    #lag the sources
    srcs_proc_lagged = lag_sources(n_lags, srcs_proc)
    #output file
    save_proc_sources_sinks(srcs_proc_lagged, snks_proc, out_dir)
    
        
if __name__ == '__main__':
    main()