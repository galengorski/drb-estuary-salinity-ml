# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 10:01:14 2022

@author: ggorski
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px 
import seaborn as sns
import TEpython_ParallelNAN2

def lag_sources_df(n_lags, srcs_df):
    n_lags_int = int(n_lags)
    '''Takes in the list of sources called srcs_list in which each list item is a 
    dataframe of site variables, and creates lagged time series for all variables
    according to the n_lag variable. Column headers are updted with the n_lag. n_lag refers
    to the number of time steps to lag, and it is agnostic of the time resolution of the data.
    length of data > n_lag >=0'''
    lagged_df = pd.DataFrame()
    for col_name in list(srcs_df.columns):
        for lag in range(0, int(n_lags_int+1)):
            #create the lagged time series and name the columns
            lagged_df[col_name+'_lag_'+str(lag)] = srcs_df[col_name].shift(lag)
    #sort the varaibles for ease of plotting
    lagged_df = lagged_df.sort_index(axis=1)
    return lagged_df

def create_correlation_matrix(sources_lagged, sinks, start_date, end_date):
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    srcs_snks = sinks.join(sources_lagged, how = 'outer')
    srcs_snks.index = pd.to_datetime(srcs_snks.index)
    srcs_snks_clipped = srcs_snks[start_date_str:end_date_str]
    snks_cols = list(sinks.columns)
    correlation_matrix = srcs_snks_clipped.corr().filter(snks_cols).drop(snks_cols).transpose()
    return(correlation_matrix)

def generate_correlation_heatmap(sources, sinks, start_date, end_date, n_lags, mask_threshold):
    #lag sources
    sources_lagged = lag_sources_df(n_lags, sources)
    #create correlation matrix
    correlation_matrix = create_correlation_matrix(sources_lagged, sinks, start_date, end_date)
    #plot heat map
    mask = abs(correlation_matrix) < mask_threshold
    plt.figure(figsize = (5,10))
    cbar_kws = {"shrink":0.85,
            'extendfrac':0.1,
            'label': 'correlation'
           }
    heatmap = sns.heatmap(correlation_matrix.transpose(), vmin = -1, vmax = 1, cbar = True, cmap='coolwarm', 
                      annot = True, cbar_kws = cbar_kws, linewidth = 1, mask = mask.transpose())
    plt.xticks(rotation=45,rotation_mode='anchor',ha = 'right')
    plt.yticks(rotation = 'horizontal')
    heatmap.set_title('|'+'Correlation'+'| > '+str(mask_threshold)+'\n '+start_date.strftime('%Y-%m-%d')+' - '+end_date.strftime('%Y-%m-%d'), fontdict={'fontsize':14}, pad=12)
    plt.show()
    
def generate_correlation_timeseries(sources, sinks, start_date, end_date, n_lags, mask_threshold):
    #lag sources
    sources_lagged = lag_sources_df(n_lags, sources)
    #create correlation matrix
    correlation_matrix = create_correlation_matrix(sources_lagged, sinks, start_date, end_date)

    
    corr_long = correlation_matrix.stack().reset_index()
    corr_long.columns = ['sink','source','correlation']
    corr_long[['site','var','none','lag']] = corr_long['source'].str.split('_',expand = True)
 
    fig = px.line(corr_long, x='lag', y='correlation', color='site', facet_row = 'var', facet_col = 'sink',
                  title="Lagged Correlation")
    fig.update_yaxes(matches=None)
    fig.show()


def create_mutual_information_matrix(sources_lagged, sinks, start_date, end_date):
    '''Takes in the lagged sources and calculates the mutual information between them and the
    sinks. mutual information is used'''
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    
    sources_lagged.index = pd.to_datetime(sources_lagged.index)
    sinks.index = pd.to_datetime(sinks.index)
    
    sources_clipped = sources_lagged[start_date_str:end_date_str]
    sinks_clipped = sinks[start_date_str:end_date_str]
    
    bins  = [11,11,11]
    dfs = pd.DataFrame()  
    MI_array = np.zeros((sources_clipped.shape[1], sinks_clipped.shape[1]))
    #MI_crit_array = np.zeros((sources_clipped.shape[1], sinks_clipped.shape[1]))
    for i,src_name in enumerate(sources_clipped):
        #print(src_name)
        for j,snk_name in enumerate(sinks_clipped):
            temp_src = sources_clipped[src_name]
            temp_snk = sinks_clipped[snk_name]
            paired = temp_src.to_frame().join(temp_snk).to_numpy()
            MI, n = TEpython_ParallelNAN2.mutinfo_new(paired, nbins = bins)
            #MIcrit = TEpython_ParallelNAN2.mutinfo_crit_newPar(paired, nbins = bins,  alpha = 0.05, numiter = 1000 ,ncores = 8)

            MI_array[i,j] = MI
            #MI_crit_array[i,j] = MIcrit[1]
            #print(str(i)+' | '+str(j)+' | '+src_name+' | '+snk_name+'| MI = '+str(MI))
    mat = pd.DataFrame(MI_array.T, columns = sources_clipped.columns)
    mat = mat.set_index(sinks_clipped.columns)

    dfs = dfs.append(mat)
    
    return(dfs)
        
#%%
###

def standardize(sr):
    standardized = list()
    for i in range(len(sr)):
        #standardize
        value_z = (sr[i] - np.nanmean(sr))/np.nanstd(sr, ddof = 1)
        standardized.append(value_z)
    return standardized

def normalize(sr):
    normalized = list()
    for i in range(len(sr)):
        #normalize
        value_z = (sr[i]-np.nanmin(sr))/(np.nanmax(sr)-np.nanmin(sr))
        normalized.append(value_z)
    return normalized

def remove_seasonal_signal(sr, sr_historical):
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

#%%
def preprocess_data_for_it(sources, sinks):
    sources_proc_df = pd.DataFrame().reindex_like(sources)
    sinks_proc_df = pd.DataFrame().reindex_like(sinks)
    
    for src in sources:
        temp_data = remove_seasonal_signal(sources[src], sources[src])
        temp_data = normalize(temp_data)
        sources_proc_df[src] = temp_data

    for snk in sinks:
        temp_data = remove_seasonal_signal(sinks[snk], sinks[snk])
        temp_data = normalize(temp_data)
        sinks_proc_df[snk] = temp_data
        
    return sources_proc_df, sinks_proc_df
        
#%%       
def generate_mutual_information_heatmap(sources, sinks, start_date, end_date, n_lags, mask_threshold):
    #preprocess data
    sources_proc, sinks_proc = preprocess_data_for_it(sources, sinks)
    
    #lag sources
    sources_lagged_proc = lag_sources_df(n_lags, sources_proc)
    #create correlation matrix
    mi_matrix = create_mutual_information_matrix(sources_lagged_proc, sinks_proc, start_date, end_date)
    #plot heat map
    mask = abs(mi_matrix) < mask_threshold
    plt.figure(figsize = (5,10))
    cbar_kws = {"shrink":0.85,
            'extendfrac':0.1,
            'label': 'mutual information'
           }
    heatmap = sns.heatmap(mi_matrix.transpose(), vmin = 0, vmax = 1, cbar = True, cmap='coolwarm', 
                      annot = True, cbar_kws = cbar_kws, linewidth = 1, mask = mask.transpose())
    plt.xticks(rotation=45,rotation_mode='anchor',ha = 'right')
    plt.yticks(rotation = 'horizontal')
    heatmap.set_title('|'+'Mutual Information'+'| > '+str(mask_threshold)+'\n '+start_date.strftime('%Y-%m-%d')+' - '+end_date.strftime('%Y-%m-%d'), fontdict={'fontsize':14}, pad=12)
    plt.show()
    
def generate_mutual_information_timeseries(sources, sinks, start_date, end_date, n_lags, mask_threshold):
    sources_proc, sinks_proc = preprocess_data_for_it(sources, sinks)

    #lag sources
    sources_lagged_proc = lag_sources_df(n_lags, sources_proc)
    #create correlation matrix
    mi_matrix = create_mutual_information_matrix(sources_lagged_proc, sinks_proc, start_date, end_date)

    
    mi_long = mi_matrix.stack().reset_index()
    mi_long.columns = ['sink','source','mutual_information']
    mi_long[['site','var','none','lag']] = mi_long['source'].str.split('_',expand = True)
 
    fig = px.line(mi_long, x='lag', y='mutual_information', color='site', facet_row = 'var', facet_col = 'sink',
                  title="Lagged Mutual Information")
    fig.update_yaxes(matches=None)
    fig.show()

#%%
# import plotly.express as px 
# import plotly.io as pio
# pio.renderers.default='browser'
#  start_date = pd.to_datetime('2015-01-01')
#  end_date = pd.to_datetime('2019-01-01')

# # generate_correlation_timeseries(sources, sinks, start_date, end_date, n_lags)
# # data.columns.to_series().str.contains(srcs[0]).any(): 
    
    
# corr_long = correlation_matrix.stack().reset_index()
# corr_long.columns = ['sink','source','correlation']
# corr_long[['site','var','none','lag']] = corr_long['source'].str.split('_',expand = True)

# fig = px.line(corr_long, x='lag', y='correlation', color='site', facet_row = 'var', facet_col = 'sink', height=600, width=800,
#               title="Lagged Correlation")
# fig.update_yaxes(matches=None)
# fig.show()

  #preprocess data
# sources_proc, sinks_proc = preprocess_data_for_it(sources, sinks)

# #lag sources
# sources_lagged_proc = lag_sources_df(n_lags, sources_proc)
# #create correlation matrix
# mi_matrix = create_mutual_information_matrix(sources_lagged_proc, sinks_proc, start_date, end_date)
