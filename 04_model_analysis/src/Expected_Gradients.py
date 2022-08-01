# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 12:26:48 2022

@author: ggorski
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
from LSTMDA_torch import LSTMDA, fit_torch_model, rmse_masked
import seaborn as sns
import sys
sys.path.append('03b_model/src')
import torch
import run_model
import yaml

#%%
device = 'cpu'
with open("04_model_analysis/model_config.yaml", 'r') as stream:
    config = yaml.safe_load(stream)

inputs = config['inputs']
target = config['target']

out_dir = config['out_dir']
run_id = config['run_id']

train_start_date = config['train_start_date']
train_end_date = config['train_end_date']
val_start_date = config['val_start_date']
val_end_date = config['val_end_date']
test_start_date = config['test_start_date']
test_end_date = config['test_end_date']

seq_len = config['seq_len']
offset = config['offset']
n_epochs_pre = config['n_epochs_pre']
hidden_units = config['hidden_units']
learn_rate_pre = config['learn_rate_pre']
recur_dropout = config['recur_dropout']
dropout = config['dropout']

#%%
inputs_xarray, target_xarray = run_model.select_inputs_targets(inputs, target, train_start_date, test_end_date, out_dir) 

prepped_model_io_data = run_model.prep_input_target_data(inputs_xarray, target_xarray, train_start_date, train_end_date, 
                       val_start_date, val_end_date, test_start_date, test_end_date, 
                       seq_len, offset, out_dir, return_data = True)

  
n_batch, seq_len, n_feat  = prepped_model_io_data['train_features'].shape

pretrain_model = LSTMDA(n_feat, hidden_units, recur_dropout, dropout)

pretrain_model.load_state_dict(torch.load(os.path.join(out_dir,run_id,'weights.pt'))) # ensure that dropout layers are active

y_pred, _ = pretrain_model(prepped_model_io_data['trainval_features'])


#%%Permutation feature importance
y_obs = prepped_model_io_data['trainval_targets']
x_data_in = prepped_model_io_data['trainval_features']
x_vars = inputs

def calc_permutation_feature_importance(model, x_data_in, x_vars, y_obs):
    original_y_hat, _ = pretrain_model(x_data_in)
    original_y_hat = original_y_hat.detach()
    rmse_original = rmse_masked(y_obs,original_y_hat)
    fi_ls=[]
    for var in range(len(x_vars)):
        x_hypothesis = x_data_in.detach().clone()
        var_range = torch.quantile(x_hypothesis[:,:,var].flatten(),torch.tensor([.1,.9]))
        #Make random distribution within the range of the target variable
        x_hypothesis[:, :, var] = (var_range[0]-var_range[1])*torch.rand_like(x_hypothesis[:, :, var])+var_range[1]
        y_hypothesis,_ = pretrain_model(x_hypothesis)
        y_hypothesis = y_hypothesis.detach()
        rmse_hypothesis = rmse_masked(y_obs,y_hypothesis)
        delta_rmse = rmse_hypothesis-rmse_original
        fi_ls.append(delta_rmse)
    return np.array(fi_ls)


#%%
feat_imp = calc_permutation_feature_importance(pretrain_model, x_data_in, x_vars, y_obs)
ordered_importance_of_vars = np.argsort(feat_imp)

### Plot up the results in order of importance
plt.figure(figsize=(4,4),constrained_layout=True) 
plt.bar(np.linspace(0,len(feat_imp),len(feat_imp)),feat_imp[ordered_importance_of_vars])
plt.xticks(np.linspace(0,len(feat_imp),len(feat_imp)), labels = np.array(x_vars)[ordered_importance_of_vars], rotation = 45,ha='right')
plt.title('Feature Importance')
plt.ylabel('∆ RMSE')
plt.xlabel("Variables");

#%%Expected gradients
def expected_gradients_lstm(x_data_in, model, n_samples, temporal_focus=None):

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

#%%
n_samples = 200
EGs_overall = expected_gradients_lstm(x_data_in, pretrain_model, n_samples, temporal_focus=None)
#%%
dates = pd.date_range(start = train_start_date, periods = x_data_in.shape[0]*x_data_in.shape[1], freq = 'D')
EGs_overall_full = np.resize(EGs_overall, (EGs_overall.shape[0]*EGs_overall.shape[1], len(x_vars)))
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
fig, ax = plt.subplots(11,1,figsize=(12,12),constrained_layout=True)
fig.suptitle("Expected Gradient with Regards to All Predictions")
for i in range(len(x_vars)):
    ax[i].plot(dates,EGs_overall_full[:, i], label = x_vars[i], color = colors[i])
    ax[i].set_title(x_vars[i])
    ax[i].plot(dates,np.repeat(0,len(dates)),'--',alpha=.3)
ax[3].set_ylabel('Expected Gradients')


#%%
#unnormalize salt front data
sf_mean = prepped_model_io_data['means_stds']['y_mean_trnval']['saltfront_daily'].data
sf_sd = prepped_model_io_data['means_stds']['y_std_trnval']['saltfront_daily'].data
tq_mean = prepped_model_io_data['means_stds']['x_mean_trnval']['discharge_01463500'].data
tq_sd = prepped_model_io_data['means_stds']['x_std_trnval']['discharge_01463500'].data


eg_df = pd.DataFrame(EGs_overall_full)
eg_df = eg_df.set_index(dates)
eg_df.columns = ['Discharge_Trenton','Discharge_Schuylkill','Water_Level_Range','Water_Level_Max','Water_Level_Resid',
                 'Water_Level_Filtered','Air_Pressure','Temperature','Wind_Direction','Wind_Speed','Wind_Direction_Speed']
eg_df['Salt_Front_Location'] = (np.resize(prepped_model_io_data['trainval_targets'], (EGs_overall.shape[0]*EGs_overall.shape[1], 1))*sf_sd)+sf_mean
eg_df['Salt_Front_Location_Preds'] = (np.resize(y_pred.detach().numpy(), (EGs_overall.shape[0]*EGs_overall.shape[1], 1))*sf_sd)+sf_mean
eg_df['Trenton_Discharge'] = (np.resize(prepped_model_io_data['trainval_features'][:,:,0], (EGs_overall.shape[0]*EGs_overall.shape[1], 1))*tq_sd)+tq_mean

#%%
#plot a single year
eg_ann = eg_df.loc['2002']
#colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = sns.color_palette('hls',12)
fig, ax = plt.subplots(4,1,figsize=(10,12),constrained_layout=True)
fig.suptitle("Expected Gradient with Regards to All Predictions")
#Discharge
for i in range(0,2):
    ax[0].plot(eg_ann.index,eg_ann.iloc[:, i], label = eg_df.columns[i], color = colors[i])
    ax[0].set_title('Discharge')
    ax[0].plot(eg_ann.index,np.repeat(0,len(eg_ann.index)),'--',alpha=.3)
    ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
#Water level
for i in range(2,6):
    ax[1].plot(eg_ann.index,eg_ann.iloc[:, i], label = eg_df.columns[i], color = colors[i])
    ax[1].set_title('Water Level')
    ax[1].plot(eg_ann.index,np.repeat(0,len(eg_ann.index)),'--',alpha=.3)
    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
#Meteorological
for i in range(6,10):
    ax[2].plot(eg_ann.index,eg_ann.iloc[:, i], label = eg_df.columns[i], color = colors[i])
    ax[2].set_title('Meteorological')
    ax[2].plot(eg_ann.index,np.repeat(0,len(eg_ann.index)),'--',alpha=.3)
    ax[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax[3].plot(eg_ann.index, eg_ann.Salt_Front_Location, label = 'Salt Front Observed', color = 'black', marker = 'o')
ax[3].plot(eg_ann.index, eg_ann.Salt_Front_Location_Preds, label = 'Salt Front Predicted', color = 'red')
ax[3].legend(loc='upper left')
ax[3].set_title('Salt Front Location (RM)')
ax[3].set_ylabel('Salt front location (RM)')
ax[3].set_ylim([20,95])
ax31=ax[3].twinx()
ax31.plot(eg_ann.index, eg_ann.Trenton_Discharge, color = 'blue', label = 'Trenton Discharge')
ax31.tick_params(axis='y', color='blue', labelcolor='blue')
ax31.set_ylabel('Discharge (cfs)')
ax31.set_ylim([0,100000])
ax31.legend(loc = 'lower left')

ax[1].set_ylabel('Expected Gradients')

#%%Plot the gradients binned by river mile interval

eg_df['Interval'] = np.nan

eg_df.loc[eg_df.Salt_Front_Location > 82,'Interval'] = '> 82'
eg_df.loc[(eg_df.Salt_Front_Location > 78) & (eg_df.Salt_Front_Location <= 82),'Interval'] = '78-82'
eg_df.loc[(eg_df.Salt_Front_Location > 70) & (eg_df.Salt_Front_Location <= 78),'Interval'] = '70-78'
eg_df.loc[(eg_df.Salt_Front_Location > 68) & (eg_df.Salt_Front_Location <= 70),'Interval'] = '68-70'
eg_df.loc[(eg_df.Salt_Front_Location > 58) & (eg_df.Salt_Front_Location <= 68),'Interval'] = '58-68'
eg_df.loc[(eg_df.Salt_Front_Location <= 58),'Interval'] = '<= 58'



eg_df_long = pd.melt(eg_df[['Discharge_Trenton','Water_Level_Range','Wind_Speed','Interval']], id_vars = ['Interval'], value_vars = ['Discharge_Trenton','Water_Level_Range','Wind_Speed'])

category_order = ['<= 58','58-68','68-70','70-78','78-82','> 82']
g = sns.catplot(x = 'Interval', y = 'value', hue = 'variable', data = eg_df_long, kind = 'box', order = category_order)

#%% Expected gradient temporal focus
doy_names = ['244','274','305']
doy = [244,274,305]
n_samples = 200
EGs_temp = {}
EGs_temp_full = {}
eg_temp_df_list = {}

for i in range(3):
    print('Calculating expected gradients for'+str(doy[i]))
    EGs_temp[doy_names[i]] = expected_gradients_lstm(x_data_in, pretrain_model, n_samples, temporal_focus=doy[i])

    dates = pd.date_range(start = train_start_date, periods = x_data_in.shape[0]*x_data_in.shape[1], freq = 'D')
    EGs_temp_full[doy_names[i]] = np.resize([doy_names[i]], ([doy_names[i]].shape[0]*[doy_names[i]].shape[1], len(x_vars)))
    
    eg_temp_df = pd.DataFrame(EGs_temp_full[doy_names[i]])
    eg_temp_df = eg_temp_df.set_index(dates)
    
    eg_temp_df.columns = ['Discharge_Trenton','Discharge_Schuylkill','Water_Level_Range','Water_Level_Max','Water_Level_Resid',
                     'Water_Level_Filtered','Air_Pressure','Temperature','Wind_Direction','Wind_Speed','Wind_Direction_Speed']
    eg_temp_df['Salt_Front_Location'] = (np.resize(prepped_model_io_data['trainval_targets'], (EGs_temp.shape[0]*EGs_temp.shape[1], 1))*sf_sd)+sf_mean
    eg_temp_df['Salt_Front_Location_Preds'] = (np.resize(y_pred.detach().numpy(), (EGs_temp.shape[0]*EGs_temp.shape[1], 1))*sf_sd)+sf_mean
    eg_temp_df['Trenton_Discharge'] = (np.resize(prepped_model_io_data['trainval_features'][:,:,0], (EGs_temp.shape[0]*EGs_temp.shape[1], 1))*tq_sd)+tq_mean
    
    eg_temp_df_list[doy_names[i]] = eg_temp_df.copy()
#%%
year = '2002'
eg_ann_temp = []
for i in range(3):
    eg_ann_temp[doy_names[i]] = eg_temp_df_list[doy_names[i]].loc[year]
    #eg_ann_temp[eg_ann_temp.Salt_Front_Location == np.nanmax(eg_ann_temp.Salt_Front_Location.values)]



fig, ax = plt.subplots(4,1,figsize=(14,3))
fig.suptitle("Expected Gradient with Regards to the Last Day in the Sequence: "+ year)
for j in range(3):
    temp = eg_ann_temp[doy_names[j]]
    for i in range(len(x_vars)):
        ax[j].plot(temp.index, temp.iloc[:,i], label = x_vars[i])
        ax[j].set_title(doy_names[i])
        
ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax[1].plot(eg_ann_temp[doy_names[j]].index, eg_ann_temp[doy_names[j]].Salt_Front_Location, label = 'Salt Front Observed', color = 'black', marker = 'o')
ax[1].plot(eg_ann_temp[doy_names[j]].index, eg_ann_temp[doy_names[j]].Salt_Front_Location_Preds, label = 'Salt Front Predicted', color = 'red')
ax[1].legend(loc='upper left')
ax[1].set_title('Salt Front Location (RM)')
ax[1].set_ylabel('Salt front location (RM)')
ax[1].set_ylim([20,95])
ax11=ax[1].twinx()
ax11.plot(eg_ann_temp[doy_names[j]].index, eg_ann_temp[doy_names[j]].Trenton_Discharge, color = 'blue', label = 'Trenton Discharge')
ax11.tick_params(axis='y', color='blue', labelcolor='blue')
ax11.set_ylabel('Discharge (cfs)')
ax11.set_ylim([0,100000])
ax11.legend(loc = 'lower left')

