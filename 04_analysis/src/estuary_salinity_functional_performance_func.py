# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 12:54:24 2022

@author: ggorski
"""
import it_functions
import itertools
import numpy as np
import os
import pandas as pd
import xarray as xr


def find_reps(run_id):
    '''
    find the list of directories that house the replicates
    Parameters
    ----------
    run_id : str
        the run number to aggregate

    Returns
    -------
    reps : list
        list of replicates

    '''
    reps = list()
    rootdir = os.path.join('03_model/out',run_id)
    for file in os.listdir(rootdir):
        d = os.path.join(rootdir, file)
        #replicates have to be in a directory with a two character name(e.g., "02")
        if os.path.isdir(d) and len(file) == 2:
            reps.append(file)
    return reps

def create_cleaned_io_file(run_id):
    '''
    creates a cleaned csv with aggregate results from 5 replicates and merges
    with input data and outputs from COAWST model, saves csv to 03_model/out
    directory

    Parameters
    ----------
    run_id : str
        the run number to aggregate

    Returns
    -------
    None.

    '''
    reps = find_reps(run_id)
    ml_sf = pd.read_csv(os.path.join('03_model/out',run_id,reps[0],'ModelResults.csv'), parse_dates = True, index_col = 'Unnamed: 0')
    for i,rep in enumerate(reps):
        temp = pd.read_csv(os.path.join('03_model/out',run_id,rep,'ModelResults.csv'), parse_dates = True, index_col = 'Unnamed: 0')
        ml_sf['saltfront_pred_'+rep] = temp['saltfront_pred'].copy()

    ml_sf_mean = ml_sf.filter(regex='saltfront_pred_').mean(axis = 1)
    ml_sf = ml_sf.merge(ml_sf_mean.rename('ml_pred'), left_index = True, right_index = True)

    ml_sf = ml_sf[['saltfront_obs','ml_pred','train/val']]

    # read in observations
    inputs = xr.open_zarr(os.path.join(f'03_model/out/{run_id}/inputs.zarr'), consolidated=False)
    inputs_df = inputs.to_dataframe()
    
    #join ml and inputs
    ml_sf_inputs = ml_sf.join(inputs_df)

    ## read in COAWST model output
    coawst_2016 = pd.read_csv('03_model/in/COAWST_model_runs/processed/COAWST_2016_7day.csv', parse_dates = True, index_col = 'date')
    coawst_2018 = pd.read_csv('03_model/in/COAWST_model_runs/processed/COAWST_2018_7day.csv', parse_dates = True, index_col = 'date')
    coawst_2019 = pd.read_csv('03_model/in/COAWST_model_runs/processed/COAWST_2019_7day.csv', parse_dates = True, index_col = 'date')
    coawst_3_years = pd.concat([coawst_2016, coawst_2018, coawst_2019])
    
    ## 
    fp_data = ml_sf_inputs.join(coawst_3_years, how = 'outer')
    
    fp_data.to_csv(os.path.join('03_model\\out',run_id,'ML_COAWST_Results_Input_Cleaned.csv'))


def calc_it_metrics_for_pairs(data_loc, sources, sinks, years, run_id):
    '''
     Generate pairs of sources and sinks from the list for each year (or range of years) listed, 
     then calculate the transfer entropy (TE), mutual information (MI), and correlation for those pairs at
     time lags of 0 to 9 days (hard coded now). Critical values for TE and MI are calculated as well. 
     The raw data are preprocessed using the preprocessing functions in it_funcitons.py. 
     The results are then converted to a dataframe and written to the file
    Parameters
    ----------
    data_loc : str
        location of cleaned data
    sources : list
        list of column names to use as sources
    sinks : list
        list of column names to use as sinks ['saltfront_obs','ml_pred','coawst_pred']
    years : list
        list of years to analyze, years can either be a single year ('2019') or a span ('2018:2020')
    run_id : str
        the model run id

    Returns
    -------
    None.

    '''
    it_dict = {}
    
    data = pd.read_csv(data_loc, parse_dates=True, index_col = 0)    

    source_sink_pairs = list(itertools.product(sources, sinks, years))

    #import preprocessing functions
    ppf = it_functions.pre_proc_func()
    
    for pair in source_sink_pairs:
        
        source = pair[0]
        sink = pair[1]
        year = pair[2]
        
        print('Calculating IT metrics for: ',source,'->',sink, year)
        
        #check if there is a colon which represents a range of years to 
        #analyze, if not it will be a single year
        if ':' in year:
            year_range = year.split(':')
            xy_data = data.loc[year_range[0]:year_range[1]][[source,sink]]
        else:
            xy_data = data.loc[year][[source,sink]]
        
        #if there are no (few) values in the sink variable then move on
        if (xy_data[sink].count() < 100) | (xy_data[source].count() < 100):
            continue
        #different pre-processing for different variables
        #x is source y is sink
        #if multiple years then the remove_seasonal_signal function is used
        if ':' in year:
            #if discharge is the source then take the log
            if 'discharge' in source:    
                x = xy_data.iloc[:,0]
                xl10 = ppf.log10(x)
                x_rss = ppf.remove_seasonal_signal(xl10)
                x_ss = ppf.standardize(x_rss)
            else:
                x = xy_data.iloc[:,0]
                x_rss = ppf.remove_seasonal_signal(x)
                x_ss = ppf.standardize(x_rss)   

            #multiple years for salt front
            y = xy_data.iloc[:,1]
            y_rss = ppf.remove_seasonal_signal(y)
            y_ss = ppf.standardize(y_rss)
        #if we are only looking at a single year then no seasonal signal is removed
        else:
            #if we are preprocessing discharge then the log transform is taken and then the data are standardized
            if 'discharge' in source:
                x = xy_data.iloc[:,0]
                xl10 = ppf.log10(x)
                x_ss = ppf.standardize(xl10)
            else:
                #no seasonal signal removed because only one year of anlaysis
                x = xy_data.iloc[:,0]
                x_ss = ppf.standardize(x)
           #no seasonal signal removed because only one year of anlaysis
            y = xy_data.iloc[:,1]
            y_ss = ppf.standardize(y) 

            
        
        #calculate it metrics for observations
        n_lags = 10
        nbins = 11
        M = np.stack((x_ss,y_ss), axis = 1)
        Mswap = np.stack((y_ss, x_ss), axis = 1)

        x_bounds = it_functions.find_bounds(M[:,0], 0.1, 99.9)
        y_bounds = it_functions.find_bounds(M[:,1], 0.1, 99.9)
        M_x_bound = np.delete(M, np.where((M[:,0] < x_bounds[0]*1.1) | (M[:,0] > x_bounds[1]*1.1)), axis = 0)
        M_xy_bound = np.delete(M_x_bound, np.where((M_x_bound[:,1] < y_bounds[0]*1.1) | (M_x_bound[:,1] > y_bounds[1]*1.1)), axis = 0)

        it_dict[pair] = it_functions.calc_it_metrics(M_xy_bound, Mswap, n_lags, nbins, calc_swap = False, alpha = 0.01, ncores = 8)
    
    #convert it_dict to data frame
    it_df = pd.DataFrame()
    for key in it_dict.keys():
        if 'TEswap' in it_dict[key].keys():
            if len(it_dict[key]['TEswap']) == 0:
                del it_dict[key]['TEswap']
                del it_dict[key]['TEcritswap']
        temp_df = pd.DataFrame(data = it_dict[key])
        temp_df['source'] = key[0]
        temp_df['sink'] = key[1]
        temp_df['year'] = key[2]
        temp_df['lag'] = range(0,len(temp_df))
        #get TE from t = 3
        it_df = pd.concat([it_df,temp_df])
            

    it_df.to_csv(os.path.join('04_analysis/out',run_id+'_it_df.csv'))

def calc_functional_performance(it_df_loc, sources, sinks, years, run_id):
    '''
    Using the calculations of transfer entropy at various time lags, this function
    calculates the functional performance for each source/sink/year pairing by 
    subtracting the modeled transfer entropy from the observed. The resulting
    dataframe of functional performances are written to the file as a csv.

    Parameters
    ----------
    it_df_loc : str
        location of the it_df dataframe created by calc_it_metrics_for_pairs
    sources : list
        list of column names to use as sources. Must be the same as found in the it_df dataframe
    sinks : list
        list of column names to use as sinks ['saltfront_obs','ml_pred','coawst_pred']. Must be the same as found in the it_df dataframe
    years : list
        list of years to analyze, years can either be a single year ('2019') or a span ('2018:2020'). Must be the same as found in the it_df dataframe
    run_id : str
        the model run id
    Returns
    -------
    None.

    '''
    it_df = pd.read_csv(it_df_loc)
    
    it_df_obs = it_df[it_df.sink == 'saltfront_obs'].groupby(['lag','year','source'])['TE'].mean()
    it_df_ml_pred = it_df[it_df.sink == 'ml_pred'].groupby(['lag','year','source'])['TE'].mean()

    ml_func_perf_df = it_df_ml_pred.subtract(it_df_obs).to_frame().reset_index()
    ml_func_perf_df['model'] = 'ml'
    
    if 'coawst_pred' in it_df.sink.unique():
        it_df_coawst_pred = it_df[it_df.sink == 'coawst_pred'].groupby(['lag','year','source'])['TE'].mean()
        coawst_func_perf_df = it_df_coawst_pred.subtract(it_df_obs).to_frame().reset_index()
        coawst_func_perf_df['model'] = 'coawst'
        
        func_perf_df = pd.concat([ml_func_perf_df, coawst_func_perf_df])
    else:
        func_perf_df = ml_func_perf_df
    
    func_perf_df = func_perf_df.rename(columns={'TE':'Func_Perf'})
    
    func_perf_df.to_csv(os.path.join('04_analysis/out',run_id+'_functional_performance_df.csv'))
    
    

def calc_functional_performance_wrapper(run_id, sources, sinks, years, it_df_loc):
    '''
    

    Parameters
    ----------
    run_id : str
        the run number to aggregate.
    data_loc : str
        location of cleaned data.
    sources : list
        list of column names to use as sources. Must be the same as found in the it_df dataframe
    sinks : list
        list of column names to use as sinks ['saltfront_obs','ml_pred','coawst_pred']. Must be the same as found in the it_df dataframe
    years : list
        list of years to analyze, years can either be a single year ('2019') or a span ('2018:2020'). Must be the same as found in the it_df dataframe
    it_df_loc : str
        location of the it_df dataframe created by calc_it_metrics_for_pairs

    Returns
    -------
    None.

    '''
    print('----------------------------------------')
    print('Prepping data...')
    print('----------------------------------------')

    create_cleaned_io_file(run_id)
    print('----------------------------------------')
    print('Calculating pair-wise it metrics...')
    print('----------------------------------------')
    
    data_loc = os.path.join('03_model/out/',run_id,'ML_COAWST_Results_Input_Cleaned.csv')

    calc_it_metrics_for_pairs(data_loc, sources, sinks, years, run_id)
    
    print('----------------------------------------')
    print('Calculating functional performance...')
    print('----------------------------------------')

    calc_functional_performance(it_df_loc, sources, sinks, years, run_id)
    
    
