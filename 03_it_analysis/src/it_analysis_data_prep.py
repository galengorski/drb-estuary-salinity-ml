# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 20:58:51 2021

@author: ggorski
"""

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
    
    srcs_list = list()
    
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
            data_col_select = data_col_select[date_start:date_end]
            #this relies on a specific file naming structure, it appends the site name to the column header
            data_col_select = data_col_select.add_suffix('_'+str(file.split('_')[2].split('.')[0]))
            print(str(file.split('_')[2].split('.')[0])+' : Sources Found')
            srcs_list.append(data_col_select)
        else:
            print(str(file.split('_')[2].split('.')[0])+' : No Data')
            continue
    return srcs_list
    
###
def select_sinks(snks, date_start, date_end):
    '''This is a filler function, for now it is hard coded and very specific to
    the salt front location spreadsheet we have, but in the future it should look
    like the select_sources function from above'''
    
    snks_list = list()

    sf_loc = pd.read_csv('methods_exploration/data/saltfront.csv', index_col = 'datetime')
    sf_loc.index = pd.to_datetime(sf_loc.index)
    sf_loc = sf_loc[date_start:date_end]
    #we'll make snks_list a list of df here they are both 2019, but these could be different model runs 
    snks_list.append(sf_loc[snks].loc['2019'])
    snks_list.append(sf_loc[snks].loc['2019'])
    
    #add a suffix to one of the snks_list entries so we can tell the difference
    snks_list[0] = snks_list[0].add_suffix('_Model_A')
    noise = np.random.gamma(8, 2, len(sf_loc))
    snks_list[0].iloc[:,0] = snks_list[0].iloc[:,0].add(noise)
    snks_list[0].iloc[:,1] = snks_list[0].iloc[:,1].add(noise)
    return snks_list

###
def save_sources_sinks(srcs_list, snks_list, out_dir):
    #create out directory if it doesn't already exist
    os.makedirs(out_dir, exist_ok = True)
    #write snks_list to file 
    snks_file = open(out_dir+'snks', "wb")
    pickle.dump(snks_list, snks_file)
    snks_file.close()
    #write srcs_list_lagged to file 
    srcs_file = open(out_dir+'srcs', "wb")
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
def create_preprocess_yaml(srcs_list, snks_list):
    '''Generates a yaml file (hard coded as 03_it_analysis/it_analysis_preprocess_config.yaml
    that contains every source and sink variable. The idea is that the yaml file will be edited
    with the preprocessing steps to take for each variable'''
    
    #read in the sources column headers
    site_var_srcs_list = list()
    for sr in range(len(srcs_list)): 
        site_var_srcs_list.append(list(srcs_list[sr].columns))
    site_var_srcs = [item for sublist in site_var_srcs_list for item in sublist]
    
    #convert sources to dictionary
    src_vals = []
    for s in range(len(site_var_srcs)):
        src_vals.extend([["none"]])
    
    #read in the sinks column headers
    site_var_snks_list = list()
    for sk in range(len(snks_list)): 
        site_var_snks_list.append(list(snks_list[sk].columns))
    site_var_snks = [item for sublist in site_var_snks_list for item in sublist]
    
    #convert sinks to dictionary
    snk_vals = []
    for s in range(len(site_var_snks)):
        snk_vals.extend([["none"]])
    
    #zip into dictionary for neat writing to yaml file add in the n_lags as that will be used
    #in the next step
    srcs_snks_dict = {'it_analysis_preprocess.py':
                 {'sources': dict(zip(site_var_srcs, src_vals)),
                 'sinks': dict(zip(site_var_snks, snk_vals)),
                 'n_lags': 1,
                 'out_dir': '03_it_analysis/out/'}}
    
    f = open('03_it_analysis/it_analysis_preprocess_config.yaml', 'w')
    f.write(yaml.dump(srcs_snks_dict))
    f.close()

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
    #number of lag days to consider for sources
    #n_lags = config['n_lags']
    #generate the sources data
    srcs_list = select_sources(srcs, date_start, date_end)    
    #generate the sinks data
    snks_list = select_sinks(snks, date_start, date_end)
    #generate the yaml file for the preprocessing steps
    create_preprocess_yaml(srcs_list, snks_list)
    #lag the sources (EDIT -- This will be moved to post pre-processing)
    #srcs_list_lagged = lag_sources(n_lags, srcs_list) (EDIT -- This will be moved to post pre-processing)
    #output file
    out_dir = config['out_dir']
    save_sources_sinks(srcs_list, snks_list, out_dir)
    
        
if __name__ == '__main__':
    main()