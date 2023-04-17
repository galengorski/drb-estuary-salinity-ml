# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 17:53:10 2022

@author: ggorski
"""
import os
import sys
sys.path.insert(0, os.path.join('03_model', 'src'))
from LSTMDA_torch import LSTMDA, rmse_masked
from matplotlib import pyplot as plt, dates
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import torch
import yaml

#%%Expected gradients
def expected_gradients_lstm(x_data_in, model, n_samples, temporal_focus=None):
    '''
    A general function for generating expected gradients from a pytorch model
    Parameters
    ----------
    x_data_in : torch.Tensor
        x variables prepared for model
    model : pytorch model
        pre trained pytorch model
    n_samples : int
        number of samples to draw for calculating expcted gradient
    temporal_focus : None (default) or int
        If the expected gradient should be calculated with respect to a single 
        day of year then that day of year should be input, otherwise none for every day. 
        The default is None.

    Returns
    -------
    numpy.ndarray of the same shape as x_data_in

    '''
    device = 'cpu'
    n_series = x_data_in.shape[0]
    #num_vars = x_set.shape[2]
    #seq_len = x_set.shape[1]
    
    for k in range(n_samples):
        # SAMPLE A SERIES FROM OUR DATA
        rand_seq = np.random.choice(n_series) # rand_time may be more accurate
        baseline_x = x_data_in[rand_seq].to(device)

        # SAMPLE A RANDOM SCALE ALONG THE DIFFERENCE
        scale = np.random.uniform()

        # SAME IG CALCULATION
        x_diff = x_data_in - baseline_x
        curr_x = baseline_x + scale*x_diff
        if curr_x.requires_grad == False:
            curr_x.requires_grad = True
        model.zero_grad()
        y,_ = model(curr_x)

        # GET GRADIENT
        if temporal_focus == None:
            gradients = torch.autograd.grad(y[:, :, :], curr_x, torch.ones_like(y[:, :, :]))
        else:
            gradients = torch.autograd.grad(y[:, temporal_focus, :], curr_x, torch.ones_like(y[:,temporal_focus, :]))

        if k == 0:
            expected_gradients = x_diff*gradients[0] * 1/n_samples
        else:
            expected_gradients = expected_gradients + ((x_diff*gradients[0]) * 1/n_samples)

    return(expected_gradients.detach().cpu().numpy())

def calc_EGs_for_model(config_loc, n_samples, write_loc):
    '''
    Uses the parameters from the model config file to prepare data 
    and models for expected_gradients_lstm function. Then read in the
    raw x variables and the model results and merge them  into a csv
    and write that csv to a directory.

    Parameters
    ----------
    config_loc : str
        location of the model config
    n_samples : int
        number of samples to draw for calculating expcted gradient
    write_loc : str
        location to write the final csv to

    Returns
    -------
    nothing

    '''
    
    
    with open(config_loc, 'r') as stream:
        config = yaml.safe_load(stream)
        
    out_dir = config['out_dir']
    run_id = config['run_id']
    train_start_date = config['train_start_date']
    hidden_units = config['hidden_units']
    recur_dropout = config['recur_dropout']
    dropout = config['dropout']
    replicates = config['replicates']

    prepped_model_io_data_file = os.path.join(out_dir,'prepped_model_io_data')

    with open(prepped_model_io_data_file, 'rb') as f:
        prepped_model_io_data = pickle.load(f)
        
    x_data_in = prepped_model_io_data['trainval_features']

    n_batch, seq_len, n_feat  = prepped_model_io_data['train_features'].shape
    model = LSTMDA(n_feat, hidden_units, recur_dropout, dropout)
    
    egs_all_reps_list = list()
    reps = range(replicates)
    for rep in reps:
        #load the model weights
        model.load_state_dict(torch.load(os.path.join(out_dir,run_id, '0'+str(rep),'weights.pt'))) 
        print('Calculating expected gradients for model in ',os.path.join(out_dir, run_id, '0'+str(rep)))
    
        egs_np_rep = expected_gradients_lstm(x_data_in, model, n_samples, temporal_focus=None)
        
        egs_all_reps_list.append(egs_np_rep)
    
    #stack the resulting list into a numpy array
    egs_all_reps_np = np.stack(egs_all_reps_list, axis = 0)
    #take the mean of all reps
    egs_np = np.mean(egs_all_reps_np, axis = 0)
    
    dates = pd.date_range(start = train_start_date, periods = x_data_in.shape[0]*x_data_in.shape[1], freq = 'D')
    EGs_overall_full = np.resize(egs_np, (egs_np.shape[0]*egs_np.shape[1], x_data_in.shape[2]))

    #read in model results and raw x_variables
    model_results = pd.read_csv(os.path.join('03_model/out',run_id,'ML_COAWST_Results_Input_Cleaned.csv'), parse_dates = True, index_col = 0)
    wl = pd.read_csv('02_munge/out/daily_summaries/noaa_nos_8557380.csv', index_col='datetime', parse_dates=True)
    met = pd.read_csv('02_munge/out/D/noaa_nerrs_delsjmet.csv', index_col = 'datetime', parse_dates = True)
    t_q = pd.read_csv('02_munge/out/D/usgs_nwis_01463500.csv', index_col = 'datetime', parse_dates = True)
    s_q = pd.read_csv('02_munge/out/D/usgs_nwis_01474500.csv', index_col = 'datetime', parse_dates = True)
    
    #join them together
    x_var_df = wl.drop(columns = 'air_pressure').join(met.drop(columns = ['precipitation', 'wind_speed_direction']), )
    x_var_df = x_var_df.join(t_q.add_suffix('_01463500')['discharge_01463500'])
    x_var_df = x_var_df.join(s_q.add_suffix('_01474500')['discharge_01474500'])

    mod_x_var = model_results.join(x_var_df.add_suffix('_obs'))
    #add the expected gradients
    eg_df = pd.DataFrame(EGs_overall_full)
    eg_df = eg_df.set_index(dates)
    eg_df.columns = ['Discharge_Trenton','Discharge_Schuylkill','Water_Level_Range','Water_Level_Max','Water_Level_Resid',
                     'Water_Level_Filtered','Air_Pressure','Temperature','Wind_Direction','Wind_Speed']
    eg_vars_df = eg_df.join(mod_x_var)

    eg_vars_df.to_csv(write_loc)
    
def plot_EGs_single_year(eg_df_loc, year, save_loc, ext):
    '''
    save a plot of the expected gradients plotted in time series
    with the observed and modeled salt front and discharge

    Parameters
    ----------
    eg_df_loc : str
        location of the expected gradients csv
    year : str
        year to plot, plotting a single calendar year at a time
    save_loc : str
        where should the plot be saved as a pdf
    ext : str
        what extension to the save the plot as usually pdf or jpg

    Returns
    -------
    None.

    '''
    egs_df = pd.read_csv(eg_df_loc, parse_dates = True, index_col = 0)
    
    eg_ann = egs_df.loc[year]
    labels = ['Trenton','Schuylkill','Range','Max','Residual','Filtered','Air pressure','Temperature','Wind direction','Wind speed']

    #colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = sns.color_palette('hls',12)
    sns.set(font_scale=1)
    sns.set_style('ticks', {'axes.linewidth': 0.5})
    plt.rcParams['xtick.major.size'] = 3
    plt.rcParams['xtick.major.width'] = 1
    plt.rcParams['ytick.major.size'] = 3
    plt.rcParams['ytick.major.width'] = 1

    plt.rcParams['xtick.bottom'] = True
    plt.rcParams['ytick.left'] = True
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['legend.frameon'] = False

    fig, ax = plt.subplots(4,1,figsize=(8,10),constrained_layout=True)

    #Discharge and salt front record
    ax[0].plot(eg_ann.index, eg_ann.saltfront_obs, label = 'Salt Front Observed', color = 'lightgray', marker = 'o', markeredgecolor = 'black', alpha = 0.4, markersize = 6)
    ax[0].plot(eg_ann.index, eg_ann.ml_pred, label = 'Salt Front Predicted', color = 'red')
    ax[0].legend(loc='upper left', bbox_to_anchor=(1.1, 0.75))
    ax[0].set_title('Salt Front Location')
    ax[0].set_ylabel('Salt front location (RM)')
    ax[0].set_ylim([20,95])
    ax01=ax[0].twinx()
    ax01.plot(eg_ann.index, eg_ann.discharge_01463500_obs, color = 'blue', label = 'Trenton Discharge')
    ax01.plot(eg_ann.index, eg_ann.discharge_01474500_obs, color = 'green', label = 'Schuylkill Discharge')
    ax01.tick_params(axis='y', color='blue', labelcolor='blue')
    ax01.set_ylim([0,100000])
    ax01.legend(loc = 'lower left', bbox_to_anchor=(1.1, 0.15))


    #Discharge
    for i in range(0,2):
        ax[1].plot(eg_ann.index,eg_ann.iloc[:, i], label = labels[i], color = colors[i])
    ax[1].plot(eg_ann.index,np.repeat(0,len(eg_ann.index)),'--',alpha=.3)
    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), title = 'Discharge')
    ax[1].set_ylim(-3,3)
    ax[1].tick_params(direction="in")

    #Water level
    for i in range(2,6):
        ax[2].plot(eg_ann.index,eg_ann.iloc[:, i], label = labels[i], color = colors[i])
    ax[2].plot(eg_ann.index,np.repeat(0,len(eg_ann.index)),'--',alpha=.3)
    ax[2].legend(loc='center left', bbox_to_anchor=(1, 0.5), title = 'Water level')
    ax[2].set_ylim(-3,3)
    ax[2].set_ylabel('Expected Gradients')

    #Meteorological
    for i in range(6,10):
        ax[3].plot(eg_ann.index,eg_ann.iloc[:, i], label = labels[i], color = colors[i])
    ax[3].plot(eg_ann.index,np.repeat(0,len(eg_ann.index)),'--',alpha=.3)
    ax[3].legend(loc='center left', bbox_to_anchor=(1, 0.5), title = 'Meteorological')
    ax[3].set_ylim(-3,3)

    #remove x axis labels for all except bottom row
    plt.setp(ax[0].get_xticklabels(), visible=False)
    plt.setp(ax[1].get_xticklabels(), visible=False)
    plt.setp(ax[2].get_xticklabels(), visible=False)

    plt.savefig(save_loc+'_'+year+'.'+ext)
    
def calc_Egs_for_all_days(config_loc, rep, n_samples, write_loc):
    
    '''
    Uses the parameters from the model config file to prepare data 
    and models for expected_gradients_lstm function with temporal
    focus for each day of the year (doy). Saves the expected gradients
    merged with model results and raw input variables in a dataframe 
    as a pickle file in write_loc

    Parameters
    ----------
    config_loc : str
        location of the model config
    rep : int
        replicate number to evalute EGs for (0-4 for salt front model)
    n_samples : int
        number of samples to draw for calculating expcted gradient
    write_loc : str
        location to write the final csv to

    Returns
    -------
    nothing

    '''
    
    doy = range(365)
    doy_names = [str(x) for x in doy]
    
    with open(config_loc, 'r') as stream:
        config = yaml.safe_load(stream)
        
    out_dir = config['out_dir']
    run_id = config['run_id']
    train_start_date = config['train_start_date']
    hidden_units = config['hidden_units']
    recur_dropout = config['recur_dropout']
    dropout = config['dropout']

    prepped_model_io_data_file = os.path.join(out_dir,'prepped_model_io_data')

    with open(prepped_model_io_data_file, 'rb') as f:
        prepped_model_io_data = pickle.load(f)
        
    x_data_in = prepped_model_io_data['trainval_features']

    n_batch, seq_len, n_feat  = prepped_model_io_data['train_features'].shape
    model = LSTMDA(n_feat, hidden_units, recur_dropout, dropout)
    
    model.load_state_dict(torch.load(os.path.join(out_dir,run_id, '0'+str(rep),'weights.pt'))) # ensure that dropout layers are active
    
    EGs_all_days = {}
    for i in range(len(doy)):
        print('Calculating expected gradients for '+str(doy[i]))
        EGs_all_days[doy_names[i]] = expected_gradients_lstm(x_data_in, model, n_samples, temporal_focus=doy[i])

    eg_df_list = {}
    EGs_temp_full = {}
    #read in model results and raw x_variables
    model_results = pd.read_csv(os.path.join('03_model/out',run_id,'ML_COAWST_Results_Input_Cleaned.csv'), parse_dates = True, index_col = 0)
    t_q = pd.read_csv('02_munge/out/D/usgs_nwis_01463500.csv', index_col = 'datetime', parse_dates = True)
    s_q = pd.read_csv('02_munge/out/D/usgs_nwis_01474500.csv', index_col = 'datetime', parse_dates = True)

    for i, day in enumerate(EGs_all_days.keys()):
        dates = pd.date_range(start = train_start_date, periods = x_data_in.shape[0]*x_data_in.shape[1], freq = 'D')
        EGs_temp_full[day] = np.resize(EGs_all_days[day], (EGs_all_days[day].shape[0]*EGs_all_days[day].shape[1], x_data_in.shape[2]))
        
        eg_temp_df = pd.DataFrame(EGs_temp_full[day])
        eg_temp_df = eg_temp_df.set_index(dates)
        
        eg_temp_df.columns = ['Discharge_Trenton','Discharge_Schuylkill','Water_Level_Range','Water_Level_Max','Water_Level_Resid',
                         'Water_Level_Filtered','Air_Pressure','Temperature','Wind_Direction','Wind_Speed']
        eg_temp_df['Salt_Front_Location'] = model_results['saltfront_obs']
        eg_temp_df['Salt_Front_Location_Preds'] = model_results['ml_pred']
        eg_temp_df['Trenton_Discharge'] = t_q.add_suffix('_01463500').discharge_01463500
        eg_temp_df['Schuylkill_Discharge'] = s_q.add_sffix('_01474500').discharge_01474500

        eg_df_list[day] = eg_temp_df.copy()
        
    with open(os.path.join(write_loc), 'wb') as handle:
        pickle.dump(eg_df_list, handle)





def EGs_lookback_df(eg_all_days_loc, t):
    '''
    Calculates the number of days it takes to accumulate t of today's gradient
    for each day of the year, where t is a threshold
    Parameters
    ----------
    eg_all_days_loc : str
        location of the dataframe of all EGs for all days, created
        by the calc_Egs_for_all_days function
    t : float
        between 0-1, this function calculates the amount of time it takes to accumulate the
        t fraction of the total gradient for each day (usually set to 0.9 or 0.99)

    Returns
    -------
    thresh_df : pandas dataframe with row for each doy and columns for each input variable,
    the number of days to accumulate gradient for each input variable are the values

    '''
    with open(eg_all_days_loc,'rb') as f:
        eg_df_dict = pickle.load(f)
    
    doy = range(365)
    doy_names = [str(x) for x in range(365)]
    all_days_thresh = {}
    for i, doy in enumerate(doy_names):
        doy_avg = np.abs(eg_df_dict[doy]).groupby([eg_df_dict[doy].index.month, eg_df_dict[doy].index.day]).mean()
        doy_avg['date'] = pd.date_range(start = '2000-01-01', end = '2000-12-31', freq = 'D')
        doy_avg = doy_avg.set_index('date')
        #doy_avg['doy'] = doy_avg.index.strftime('%j')
        doy_avg = doy_avg.drop(columns = ['Trenton_Discharge','Schuylkill_Discharge', 'Salt_Front_Location', 'Salt_Front_Location_Preds'])
        doy_cumsum = doy_avg.cumsum()
        thresh = doy_cumsum.iloc[i,:]*(1-t)
        thresh_dict = {}
        for j in range(len(thresh)):
            thresh_dict[doy_cumsum.columns[j]] = int(doy_cumsum.index[i].strftime('%j'))- int(doy_cumsum[doy_cumsum.iloc[:,j] >= thresh[j]].index[0].strftime('%j'))

        all_days_thresh[doy] = thresh_dict

    thresh_df = pd.DataFrame(data = all_days_thresh).transpose()
    thresh_df['date'] = pd.date_range(start = '2000-01-01', end = '2000-12-30', freq = 'D')
    thresh_df = thresh_df.set_index('date')
    
    return thresh_df



def plot_EGs_lookback(eg_all_days_loc, t, save_loc):
    '''
    Take the lookback csv created in EGs_lookback_df and 
    generate and save a plot

    Parameters
    ----------
   eg_all_days_loc : str
       location of the dataframe of all EGs for all days, created
       by the calc_Egs_for_all_days function
   t : float
       between 0-1, this function calculates the amount of time it takes to accumulate the
       t fraction of the total gradient for each day (usually set to 0.9 or 0.99)
    save_loc : save location for the plot

    Returns
    -------
    None.

    '''
    
    thresh_df = EGs_lookback_df(eg_all_days_loc, t)
    
    thresh_df = thresh_df[thresh_df.index > '2000-03-01']    
    t_pct = t*100
    sns.set(font_scale=0.95)
    sns.set_style('ticks', {'axes.linewidth': 0.5})
    plt.rcParams['xtick.major.size'] = 3
    plt.rcParams['xtick.major.width'] = 1
    plt.rcParams['ytick.major.size'] = 3
    plt.rcParams['ytick.major.width'] = 1

    plt.rcParams['xtick.bottom'] = True
    plt.rcParams['ytick.left'] = True
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['legend.handlelength'] = 2
    dateform = dates.DateFormatter('%b-%d')
    
    colors = sns.color_palette('hls',12)
    #sns.set_style('white')
    fig= plt.figure(figsize=(10,12))#,constrained_layout=True, sharey = True)
    fig.suptitle(f"How many days does it take to accumulate {t_pct}% of the expected gradient for each day?")
    
    #Discharge
    ax0 = plt.subplot(311)
    for i in range(0,2):        
        ax0.plot(thresh_df.iloc[:, i], label = thresh_df.columns[i], color = colors[i])
    ax0.set_title('Discharge')
    ax0.legend()
    ax0.xaxis.set_major_formatter(dateform)
    ax0.xaxis.set_major_locator(dates.MonthLocator(interval = 1))
    ax0.tick_params(direction='in')
    
    #Water level
    ax1 = plt.subplot(312, sharey = ax0)
    for i in range(2,6):
        ax1.plot(thresh_df.iloc[:, i], label = thresh_df.columns[i], color = colors[i])
    ax1.set_title('Water Level')
    ax1.legend()
    ax1.set_ylabel(f'Number of days to accumulate\n{t_pct}% of expected gradient')
    ax1.xaxis.set_major_formatter(dateform)
    ax1.xaxis.set_major_locator(dates.MonthLocator(interval = 1))
    ax1.tick_params(direction="in")
    
    #Meteorological
    ax2 = plt.subplot(313, sharey = ax0)
    for i in range(6,10):
        ax2.plot(thresh_df.iloc[:, i], label = thresh_df.columns[i], color = colors[i])
    ax2.set_title('Meteorological')
    ax2.legend()
    ax2.xaxis.set_major_formatter(dateform)
    ax2.xaxis.set_major_locator(dates.MonthLocator(interval = 1))
    ax2.tick_params(direction="in")
    plt.savefig(save_loc)

def calc_permutation_feature_importance(model, x_data_in, x_vars, y_obs):
    '''
    Calculate permutation feature importance for all variables in x_vars.
    Feature importance is calculated as the difference in RMSE when the variable
    values are replaced with a random distribution within the range of the variable values
    Parameters
    ----------
    model : pytorch model
        pre trained pytorch model
    x_data_in : torch.Tensor
        x variables prepared for model
    x_vars : list
        list of names of x_variables to make calculations for
    y_obs : torch.tensor
        y observations in model ready form

    Returns
    -------
    np.array
        dimensions len(x_vars) with each entry a difference in RMSE

    '''
    original_y_hat, _ = model(x_data_in)
    original_y_hat = original_y_hat.detach()
    rmse_original = rmse_masked(y_obs,original_y_hat)
    fi_ls=[]
    for var in range(len(x_vars)):
        x_hypothesis = x_data_in.detach().clone()
        var_range = torch.quantile(x_hypothesis[:,:,var].flatten(),torch.tensor([.1,.9]))
        #Make random distribution within the range of the target variable
        x_hypothesis[:, :, var] = (var_range[0]-var_range[1])*torch.rand_like(x_hypothesis[:, :, var])+var_range[1]
        y_hypothesis,_ = model(x_hypothesis)
        y_hypothesis = y_hypothesis.detach()
        rmse_hypothesis = rmse_masked(y_obs,y_hypothesis)
        delta_rmse = rmse_hypothesis-rmse_original
        fi_ls.append(delta_rmse)
    return np.array(fi_ls)

def calc_feature_importance_reps(config_loc, reps, write_loc):
    '''
    A wrapper function which uses the parameters from the model 
    config file to prepare data and models for 
    calc_permutation_feature_importance function. Then make the calculations 
    and take the mean and standard deviation of the replicates and 
    collect them in a csv and save it to write_loc

    Parameters
    ----------
    config_loc : str
        location of the model config
    reps : list
        list of strings for estuary salinity ['00','01','02','03','04']
    write_loc : str
        location to write the final csv to

    Returns
    -------
    nothing

    '''
    
    with open(config_loc, 'r') as stream:
        config = yaml.safe_load(stream)
        
    out_dir = config['out_dir']
    run_id = config['run_id']
    hidden_units = config['hidden_units']
    recur_dropout = config['recur_dropout']
    dropout = config['dropout']
    x_vars = config['inputs']

    prepped_model_io_data_file = os.path.join(out_dir,'prepped_model_io_data')

    with open(prepped_model_io_data_file, 'rb') as f:
        prepped_model_io_data = pickle.load(f)
        
    x_data_in = prepped_model_io_data['trainval_features']

    n_batch, seq_len, n_feat  = prepped_model_io_data['train_features'].shape
   
    y_obs = prepped_model_io_data['trainval_targets']
    
    replicate_feat_imp = np.empty([10,len(reps)])

    for i,rep in enumerate(reps):
        print('Calculating feature importance for replicate ', rep)
        model = LSTMDA(n_feat, hidden_units, recur_dropout, dropout)
        model.load_state_dict(torch.load(os.path.join(out_dir,run_id, rep,'weights.pt'))) # ensure that dropout layers are active
        
        y_pred, _ = model(prepped_model_io_data['trainval_features'])

        temp_feat_imp = calc_permutation_feature_importance(model, x_data_in, x_vars, y_obs)

        replicate_feat_imp[:,i] = temp_feat_imp
        
    mean_feat_imp = np.mean(replicate_feat_imp, axis = 1)
    std_feat_imp = np.std(replicate_feat_imp, axis = 1)
    
    replicate_feat_imp_df = pd.DataFrame(replicate_feat_imp)
    replicate_feat_imp_df['mean_feat_imp']  = mean_feat_imp
    replicate_feat_imp_df['std_feat_imp'] = std_feat_imp
    replicate_feat_imp_df.index = ['Trenton discharge','Schuylkill discharge','Water level range','Water level max','Water level residual','Water level filtered','Air pressure','Air temperature','Wind direction','Wind speed']

    replicate_feat_imp_df.to_csv(write_loc)
    
def plot_feature_importance(feat_imp_loc, save_loc):
    '''
    read in the feature importance csv and save a barplot

    Parameters
    ----------
    feat_imp_loc : str
        location of the feature importance csv
    save_loc : str
        save location for the barplot

    Returns
    -------
    None.

    '''
    
    feat_imp = pd.read_csv(feat_imp_loc, index_col = 0)
    
    y_pos = np.arange(feat_imp.shape[0])
    ordered_mean_imp = feat_imp.sort_values(by = 'mean_feat_imp').mean_feat_imp
    y_labels = feat_imp.sort_values(by = 'mean_feat_imp').index

    fig, ax = plt.subplots()
    ax.barh(y_pos, ordered_mean_imp, align = 'center', alpha = 0.5, ecolor = 'black', capsize = 10)
    ax.set_xlabel('∆ RMSE')
    ax.set_ylabel('Variable')
    ax.set_title('Relative feature importance')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels)
    plt.tight_layout()
    plt.savefig(save_loc)
   


