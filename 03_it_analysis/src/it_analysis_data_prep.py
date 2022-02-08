# -*- coding: utf-8 -*-
#Created on Wed Dec 22 20:58:51 2021
#@author: ggorski


import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pickle
import yaml
import utils


###
def download_s3_to_local(s3_dir_prefix, local_outdir, file_id):
    '''download data files from s3 bucket to local machine for development
    file_id - a file identifier substring that is contained within all 
    the file names you want to download. For example 'usgs_nwis' will 
    download all files with 'usgs_nwis' in the file name'''
    
    # assumes we are using a credential profile names 'dev'
    write_location = 'local'
    aws_profile = 'dev'
    s3_client = utils.prep_write_location(write_location, aws_profile)
    # end the name of the bucket you want to read/write to:
    s3_bucket = 'drb-estuary-salinity'
    
    # create the output file directory on your local
    os.makedirs(local_outdir, exist_ok=True)

    # loop through all objects with this prefix that contain .csv and file_id and download
    for obj in s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=s3_dir_prefix)['Contents']:
        s3_fpath = obj['Key']
        if ".csv" and file_id not in s3_fpath:
            continue
        local_fpath = os.path.join(local_outdir,obj['Key'].split('/')[2])
        s3_client.download_file(s3_bucket, s3_fpath, local_fpath)
        print(s3_fpath+' Downloaded to local')

###
def select_sources(srcs, date_start, date_end):
    '''select the variables you are interested in examining, 
    srcs must be a list using the exact variable names,
    it will return a list of dataframes, each dataframe corresponding to a site
    with the requested variables as columns'''
    
    print('Looking for sources: ', srcs)
    
    srcs_list = list()
    srcs_list_historical = list()
    
    for file in os.listdir('02_munge/out/'):
        #read each file
        data = pd.read_csv('02_munge/out/'+file)
        #print(file)
        #print(data.head())
        sources = list()
        #if the columns of the dataframe contain any of the entries in srcs
        if data.columns.to_series().str.contains(srcs[0]).any(): 
            #select the columns that contain the entries in srcs
            for s in srcs:
                sources.append(data.loc[:,data.columns.to_series().str.contains(s)].columns[0])
                #print(sources)
           #subset those columns
            data_col_select = data.loc[:,sources]
            data_col_select = data_col_select.set_index(pd.to_datetime(data['datetime']))
            #this relies on a specific file naming structure, it appends the site name to the column header
            data_col_select = data_col_select.add_suffix('_'+str(file.split('_')[2].split('.')[0]))

            #make a copy to store for historical data
            data_col_select_historical = data_col_select[:date_end].copy()
            #store all the data in _historical for preprocessing
            srcs_list_historical.append(data_col_select_historical)
            #subset only the date range of interest
            data_col_select = data_col_select[date_start:date_end]
            print(str(file.split('_')[2].split('.')[0])+' : Sources Found')
            srcs_list.append(data_col_select)
        else:
            print(str(file.split('_')[2].split('.')[0])+' : No Data')
            continue
    return srcs_list, srcs_list_historical
    
###
def select_sinks(snks, date_start, date_end):
    '''This is a filler function, for now it is hard coded and very specific to
    the salt front location spreadsheet we have, but in the future it should look
    like the select_sources function from above'''
    
    print('Looking for sinks: ', snks)
    
    snks_list = list()
    snks_list_historical = list()

    #read in the salt front record
    sf_loc = pd.read_csv('methods_exploration/data/saltfront.csv', index_col = 'datetime')
    sf_loc.index = pd.to_datetime(sf_loc.index)
    #make a copy for historical
    sf_loc_historical = sf_loc['2015-01-01':date_end].copy()
    #save it to a list of _historical
    snks_list_historical.append(sf_loc_historical)
    snks_list_historical.append(sf_loc_historical)
    #trim sf_loc to the dates of interest
    sf_loc = sf_loc[date_start:date_end]
    
    #we'll make snks_list a list of df here they are both 2019, but these could be different model runs 
    snks_list.append(sf_loc)
    snks_list.append(sf_loc)
    
    #add a suffix to one of the snks_list entries so we can tell the difference
    snks_list[0] = snks_list[0].add_suffix('_Model_A')
    snks_list_historical[0] = snks_list_historical[0].add_suffix('_Model_A')
    #noise = np.random.gamma(8, 2, len(sf_loc))
    #snks_list[0].iloc[:,0] = snks_list[0].iloc[:,0].add(noise)
    #snks_list[0].iloc[:,1] = snks_list[0].iloc[:,1].add(noise)
    return snks_list, snks_list_historical

###
def save_sources_sinks(srcs_list, snks_list, out_dir):
    #create out directory if it doesn't already exist
    os.makedirs(out_dir, exist_ok = True)
    #write snks_list to file 
    snks_file = open(out_dir+'snks_proc', "wb")
    pickle.dump(snks_list, snks_file)
    snks_file.close()
    #write srcs_list_lagged to file 
    srcs_file = open(out_dir+'srcs_proc_lagged', "wb")
    pickle.dump(srcs_list, srcs_file)
    srcs_file.close()
    print('sinks and sources saved to file')


###
def lag_sources(n_lags, srcs_list):
    '''Takes in the list of sources called srcs_list in which each list item is a 
    dataframe of site variables, and creates lagged time series for all variables
    according to the n_lag variable. Column headers are updted with the n_lag. n_lag refers
    to the number of time steps to lag, and it is agnostic of the time resolution of the data.
    length of data > n_lag >=0'''
    srcs_list_metrics = dict.fromkeys(srcs_list.keys())
    for key in srcs_list:
        site_list = srcs_list[key]
        #lag the source variables
        for s in range(len(site_list)):
            for col_name in list(site_list[s].columns):
                for lag in range(1, n_lags+1):
                    #create the lagged time series and name the columns
                    site_list[s][col_name+'_lag_'+str(lag)] = site_list[s][col_name].shift(lag)
            #sort the varaibles for ease of plotting
            site_list[s] = site_list[s].sort_index(axis=1)
        srcs_list_metrics[key] = site_list
    return srcs_list_metrics
###
# def create_preprocess_yaml(srcs_list, snks_list):
#     '''Generates a yaml file (hard coded as 03_it_analysis/it_analysis_preprocess_config.yaml
#     that contains every source and sink variable. The idea is that the yaml file will be edited
#     with the preprocessing steps to take for each variable'''
    
#     #read in the sources column headers
#     site_var_srcs_list = list()
#     for sr in range(len(srcs_list)): 
#         site_var_srcs_list.append(list(srcs_list[sr].columns))
#     site_var_srcs = [item for sublist in site_var_srcs_list for item in sublist]
    
#     #convert sources to dictionary
#     src_vals = []
#     for s in range(len(site_var_srcs)):
#         src_vals.extend([["none"]])
    
#     #read in the sinks column headers
#     site_var_snks_list = list()
#     for sk in range(len(snks_list)): 
#         site_var_snks_list.append(list(snks_list[sk].columns))
#     site_var_snks = [item for sublist in site_var_snks_list for item in sublist]
    
#     #convert sinks to dictionary
#     snk_vals = []
#     for s in range(len(site_var_snks)):
#         snk_vals.extend([["none"]])
    
#     #zip into dictionary for neat writing to yaml file add in the n_lags as that will be used
#     #in the next step
#     srcs_snks_dict = {'it_analysis_preprocess.py':
#                  {'sources': dict(zip(site_var_srcs, src_vals)),
#                  'sinks': dict(zip(site_var_snks, snk_vals)),
#                  'n_lags': 1,
#                  'out_dir': '03_it_analysis/out/'}}
    
#     f = open('03_it_analysis/it_analysis_preprocess_config.yaml', 'w')
#     f.write(yaml.dump(srcs_snks_dict))
#     f.close()

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

    def remove_seasonal_signal(self, sr, sr_historical):
        #calculate doy for sr
        sr_doy = sr.index.strftime('%j')
        #convert sr_historical to df
        sr_historical_df = sr_historical.to_frame().copy()
        #calculate doy
        sr_historical_df['doy'] = list(sr_historical.index.strftime('%j'))
        #calculate the doy means
        doy_means = sr_historical_df.groupby('doy').mean()
        #convert the index (doy) to int64
        doy_means.index = doy_means.index.astype('int64')
        
        seasonal_removed = list()
        for i in range(len(sr)):
            doy = int(sr_doy[i])
            doy_mean = doy_means.loc[doy]
            value = sr.iloc[i]-doy_mean[0]
            seasonal_removed.append(value)
        return seasonal_removed

###
def apply_preprocessing_functions(var_list, var_list_historical, source_sink, out_dir):
    '''Apply preprocessing functions to each variable. The preprocessing functions
    and the order that they are applied in are stored in the it_analysis_preprocess_config.yaml
    file. For each step in the process it saves a histogram of that data to the out_dir.
    The available preprocessing functions are of the class pre_proc_func and can be custom
    designed within the it_analysis_preprocess.py file. This function (apply_preprocessing_functions) has
    inputs and outputs that are structurally identical. Input is a list of dataframes of 'raw' data
    and output is a list of dataframes of processed data'''

    #assert (source_sink == 'sources'|source_sink == 'sinks'),'variable source_sink must be set to "sources" or "sinks"'
    #import config
    with open("03_it_analysis/it_analysis_data_prep_config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)['it_analysis_data_prep.py']['preprocess_steps']
    
    #get the functions of class pre_proc_func
    ppf = globals()['pre_proc_func']()
    
    #make an ouptut directory for the histograms
    os.makedirs(out_dir+'preprocess_plots', exist_ok=True)
    
    #make empty dictionary to store processed data, one entry for each metric
    processed_dict = dict.fromkeys(config.keys())
    
    #for each metric
    for n_metric, metric in enumerate(config.keys()):
        #print(n_metric, metric)
        
        #select the steps for the sources and sinks        
        if source_sink == 'sources':
            pre_process_steps = config[metric]['sources']
        else:
            pre_process_steps = config[metric]['sinks']
    
        #make new empty list of processed source data
        data_proc_list = []
        #for each entry in the list, likely each site
        for site_num, site_data in enumerate(var_list):
            #print(site_num, site_data)
            #create a new data frame with the same structure as the raw data
            var_proc_df = pd.DataFrame().reindex_like(var_list[site_num])
            
            #for each preprocessing key
            for pp_key in list(pre_process_steps.keys()):
                #print(pre_process_steps[pp_key])
                cols = list(var_proc_df.columns)
                #ucn is the unique column name
                #find the ucn that contains the pp_key, should only be one per var_list[site_num]
                ucn = [string for string in cols if pp_key in string][0]
                
                #if the preprocess step is set to 'none' create a histogram of the raw data
                fig, (ax1, ax2) = plt.subplots(2)
                ax1.plot(var_list[site_num].index, var_list[site_num][ucn])
                ax1.set_ylabel(ucn)
                ax2.hist(var_list[site_num][ucn])
                ax2.set_ylabel('Count')
                ax1.set_title(ucn+'_Raw')
                fig.savefig(out_dir+'preprocess_plots/' +ucn + '_raw.png', bbox_inches = 'tight')
                #plt.show()
                plt.close()

                if pre_process_steps[pp_key][0] == 'none':                    
                    #store the variable's data
                    raw_data = var_list[site_num][ucn].copy()
                    var_proc_df[ucn] = raw_data
                    print(ucn+'-- nothing to do here')
                else:
                    #if there are preprocessing steps to do
                    #store the data in temp data
                    temp_data = var_list[site_num][ucn].copy()
                    temp_data_historical = var_list_historical[site_num][ucn].copy()
                    
                    #for each preprocessing step for that variable
                    fname = os.path.join(out_dir, 'preprocess_plots', ucn)
                    for count,value in enumerate(pre_process_steps[pp_key]):
                        #apply the function
                        func = getattr(ppf,value)
                        
                        if value == 'remove_seasonal_signal':
                            #print('got the right function')
                            temp_data = func(temp_data, temp_data_historical)
                        else:
                            temp_data = func(temp_data)
                        
                        #create histogram
                        fig, (ax1, ax2) = plt.subplots(2)
                        ax1.plot(var_list[site_num].index, temp_data)
                        ax1.set_ylabel(ucn)
                        ax2.hist(temp_data)
                        ax2.set_ylabel('Count')
                        ax1.set_title(ucn + '_' + value + '_step ' + str(count+1)+ '/'+ str(len(pre_process_steps[pp_key])))
                        fname = fname+'_'+ value
                        fig.savefig(fname+'.png', bbox_inches = 'tight')
                        #plt.show()
                        plt.close()
                        
                        #store the variable's data
                        var_proc_df[ucn] = temp_data
                        print(ucn+"-- apply: "+value)
            data_proc_list.append(var_proc_df)
        processed_dict[metric] = data_proc_list
    return(processed_dict)

###
def main():
    #import config
    with open("03_it_analysis/it_analysis_data_prep_config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)['it_analysis_data_prep.py']
    #select the sources
    srcs = config['srcs']
    #select the sinks
    snks = config['snks']
    #start date for analysis
    date_start = config['date_start']
    #end date for analysis
    date_end = config['date_end']
    #select
    
    #number of lag days to consider for sources
    n_lags = config['n_lags']
    #generate the sources data
    srcs_list, srcs_list_historical = select_sources(srcs, date_start, date_end)    
    #generate the sinks data
    snks_list, snks_list_historical = select_sinks(snks, date_start, date_end)
    #output file
    out_dir = config['out_dir'] 
    #process the sources
    srcs_proc = apply_preprocessing_functions(srcs_list, srcs_list_historical, 'sources', out_dir)
    #process the sinks
    snks_proc = apply_preprocessing_functions(snks_list, snks_list_historical, 'sinks', out_dir)

    #lag the sources
    srcs_list_lagged = lag_sources(n_lags, srcs_proc)
    out_dir = config['out_dir']
    save_sources_sinks(srcs_list_lagged, snks_proc, out_dir)
    
        
if __name__ == '__main__':
    main()