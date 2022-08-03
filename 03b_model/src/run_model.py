# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 15:44:55 2022

@author: ggorski
"""

from datetime import date
import itertools
from LSTMDA_torch import LSTMDA, fit_torch_model, rmse_masked
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
from river_dl.preproc_utils import separate_trn_tst, scale, split_into_batches
import shutil
import torch
import yaml


with open("03b_model/model_config.yaml", 'r') as stream:
    config = yaml.safe_load(stream)

inputs = config['inputs']
target = config['target']
inc_ante = config['include_antecedant_data']

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
hidden_units = config['hidden_units']
n_epochs = config['n_epochs']
learn_rate = config['learn_rate']
recur_dropout = config['recur_dropout']
dropout = config['dropout']
inc_ante = config['include_antecedant_data']


###
def select_inputs_targets(inputs, target, train_start_date, test_end_date, out_dir, inc_ante):
    '''
    select the input variables and target variables you are interested in examining, 
    saves the inputs and variables as separate .zarr files
    
    Parameters
    ----------
    inputs : list
        list of input variables used as predictors
    target : str
        target variable
    train_start_date : str
        start of training in YYYY-MM-DD format
    test_end_date : str
        end of testing in YYYY-MM-DD format
    out_dir : str
        directory where inputs and targets .zarr will be written
    inc_ante : bool
        flag to determine if antecedant conditions should be explicitly included, if True, 
        a rolling look back average of the last n days will be included as an input variable, 
        the number of days and variable are written in the config file

    Returns
    -------
    inputs_xarray : xarray.dataset
        inputs used for modeling
    target_xarray : xarray.dataset
        target variable

    '''
    
    inputs_df = pd.DataFrame()
        
    files_to_read = []
    for top,dirs,files in os.walk('02_munge/out/'):
        if top == '02_munge/out/D' or top == '02_munge/out/daily_summaries':
            files_to_read.extend([os.path.join(top,files) for files in files])
    
    for inp in inputs:
        var = '_'.join(inp.split('_')[:-1])
        site = inp.split('_')[-1]
        
         
        inp_file = [s for s in files_to_read if site in s]
        data = pd.read_csv(inp_file[0], parse_dates = True, index_col = 'datetime')
        try:
            inputs_df[inp] = data[var]
        except:
            continue
    
    #inputs_df = inputs_df.join(nerrs_input)
    inputs_df = inputs_df[train_start_date:test_end_date]
    
    if inc_ante:
        
        win_sizes = config['window_size']
        ante_var = config['antecedant_variables']
        
        #if discharge is in the antecedant variable list make a combined
        #discharge variable adding the discharge from the Delaware at Trenton (01474500)
        #to the discharge from the Schuylkill at Philadelphia (01463500)
        if 'discharge' in ante_var:
            inputs_df['discharge'] = inputs_df['discharge_01463500']+inputs_df['discharge_01474500']
        
        for win_size in win_sizes:    
            for j,col in enumerate(ante_var):
                inputs_df[col+'_'+str(win_size).rjust(3,'0')+'_mean'] = float('nan')
                col_values = []
                for i in range(len(inputs_df)-win_size):
                    temp_win = inputs_df[col].iloc[i:win_size+i].copy()
                    win_mean = np.mean(temp_win)
                    col_values.append(win_mean)
                        
                print('Calculating '+inputs_df.columns[-1]+' '+str(win_size))
                inputs_df.iloc[win_size:len(inputs_df), -1] = col_values
        #if discharge is in the antecedant variable list
        if 'discharge' in ante_var:
            #remove the combined discharge from the list of inputs
            #this way we have the discharge at Trenton and from the Schuylkill as separate predictors
            #and the antecedant discharge of the two combined as another predictor
            inputs_df = inputs_df.drop(columns = 'discharge')
        
        inputs_df = inputs_df.dropna()
    
    
    #read in the salt front record
    target_df = pd.read_csv(os.path.join('03a_it_analysis', 'in', 'saltfront.csv'), parse_dates = True, index_col = 'datetime')
    target_df = target_df['saltfront_daily'].to_frame()
    target_df.index = pd.to_datetime(target_df.index.date)
    target_df = target_df[str(inputs_df.index[0]):test_end_date]
    target_df.index = target_df.index.rename('datetime')
    
    #set everything below 54 to nan
    target_df_c = target_df.copy()
    #set river mile < 54 to NA as the salt front is not tracked below river mile 54 per DRBC
    # if train_high:
    #     mask_high = target_df_c['saltfront_daily'] < 78
    #     target_df_c.loc[mask_high,'saltfront_daily'] = np.nan
    # else:
    #     mask = target_df_c['saltfront_daily'] < 54
    #     target_df_c.loc[mask,'saltfront_daily'] = np.nan
    mask = target_df_c['saltfront_daily'] < 54
    target_df_c.loc[mask,'saltfront_daily'] = np.nan
       
    inputs_xarray = inputs_df.to_xarray()
    target_xarray = target_df_c.to_xarray()
    
    if os.path.exists(os.path.join(out_dir,'inputs.zarr')):
        shutil.rmtree(os.path.join(out_dir,'inputs.zarr'))
    if os.path.exists(os.path.join(out_dir, 'target.zarr')):
        shutil.rmtree(os.path.join(out_dir, 'target.zarr'))
    
    inputs_xarray.to_zarr(os.path.join(out_dir,'inputs.zarr'))
    target_xarray.to_zarr(os.path.join(out_dir,'target.zarr'))
    
    return inputs_xarray, target_xarray


def prep_input_target_data(inputs_xarray, target_xarray, 
                           train_start_date, train_end_date, 
                           val_start_date, val_end_date, 
                           test_start_date, test_end_date, 
                           seq_len, offset, out_dir):
    '''
    takes inputs and target variables as xarray files, then: 
    1) splits them into train, val, test sets
    2) scales each set (subtract mean, divide by stdev)
    3) splits data into batches of size [nbatches, seq_len, n_features]
    4) converts data into pytorch tensors
    5) saves input and output together to file
    Parameters
    ----------
    inputs_xarray : xarray.dataset
        full modeling inputs
    target_xarray : xarray.dataset
        full target data
    train_start_date : str
        start of training in YYYY-MM-DD format
    train_end_date : str
        end of training in YYYY-MM-DD format
    val_start_date : str
        start of validation in YYYY-MM-DD format
    val_end_date : str
        end of validation in YYYY-MM-DD format
    test_start_date : str
        start of testing in YYYY-MM-DD format
    test_end_date : str
        end of testing in YYYY-MM-DD format
    seq_len : int
        sequence length
    offset : float
        How to offset the batches. Values < 1 are taken as fractions, (e.g., 0.5 means that
        the first batch will be 0-365 and the second will be 182-547), values > 1 are used as a constant number of
        observations to offset by.
    out_dir : str
        the directory where the prepped data will be written to as a pickle file

    Returns
    -------
    None.

    '''
    
    #split into train val test sets
    y_trn, y_val, y_tst = separate_trn_tst(target_xarray,'datetime',train_start_date, train_end_date,
                                                              val_start_date, val_end_date,
                                                              test_start_date, test_end_date)
    x_trn, x_val, x_tst = separate_trn_tst(inputs_xarray,'datetime',train_start_date, train_end_date,
                                                              val_start_date, val_end_date,
                                                              test_start_date, test_end_date)
    y_trnval, _, _ = separate_trn_tst(target_xarray,'datetime',train_start_date, val_end_date,
                                                              val_start_date, val_end_date,
                                                              test_start_date, test_end_date)
    x_trnval, _, _ = separate_trn_tst(inputs_xarray,'datetime',train_start_date, val_end_date,
                                                              val_start_date, val_end_date,
                                                              test_start_date, test_end_date)
    #scale the data
    x_scl_trn, x_std_trn, x_mean_trn = scale(x_trn)
    y_scl_trn, y_std_trn, y_mean_trn = scale(y_trn)
    
    x_scl_val, x_std_val, x_mean_val = scale(x_val, std = x_std_trn, mean = x_mean_trn)
    y_scl_val, y_std_val, y_mean_val = scale(y_val, std = y_std_trn, mean = y_mean_trn)
    
    x_scl_trnval, x_std_trnval, x_mean_trnval = scale(x_trnval, std = x_std_trn, mean = x_mean_trn)
    y_scl_trnval, y_std_trnval, y_mean_trnval = scale(y_trnval, std = y_std_trn, mean = y_mean_trn)
    
    x_scl_tst, x_std_tst, x_mean_tst = scale(x_tst, std = x_std_trn, mean = x_mean_trn)
    y_scl_tst, y_std_tst, y_mean_tst = scale(y_tst, std = y_std_trn, mean = y_mean_trn)
    
    means_stds = {'x_std_trn':x_std_trn, 'x_mean_trn':x_mean_trn,
                 'y_std_trn':y_std_trn, 'y_mean_trn':y_mean_trn,
                 'x_std_val':x_std_val, 'x_mean_val':x_mean_val,
                 'y_std_val':y_std_val, 'y_mean_val':y_mean_val,
                 'x_std_trnval':x_std_trnval, 'x_mean_trnval':x_mean_trnval,
                 'y_std_trnval':y_std_trnval, 'y_mean_trnval':y_mean_trnval,
                 'x_std_tst':x_std_tst, 'x_mean_tst':x_mean_tst,
                 'y_std_tst':y_std_tst, 'y_mean_tst':y_mean_tst}
    
    #split into batches of length seq_len
    y_trn_btch = split_into_batches(y_scl_trn.to_array().to_numpy(), seq_len, offset).swapaxes(1,2)
    x_trn_btch = split_into_batches(x_scl_trn.to_array().to_numpy(), seq_len, offset).swapaxes(1,2)
    
    y_val_btch = split_into_batches(y_scl_val.to_array().to_numpy(), seq_len, offset).swapaxes(1,2)
    x_val_btch = split_into_batches(x_scl_val.to_array().to_numpy(), seq_len, offset).swapaxes(1,2)
    
    y_trnval_btch = split_into_batches(y_scl_trnval.to_array().to_numpy(), seq_len, offset).swapaxes(1,2)
    x_trnval_btch = split_into_batches(x_scl_trnval.to_array().to_numpy(), seq_len, offset).swapaxes(1,2)
    
    y_tst_btch = split_into_batches(y_scl_tst.to_array().to_numpy(), seq_len, offset).swapaxes(1,2)
    x_tst_btch = split_into_batches(x_scl_tst.to_array().to_numpy(), seq_len, offset).swapaxes(1,2)
    
    #convert to torch tensors
    train_features = torch.from_numpy(np.array(x_trn_btch)).float()
    train_targets = torch.from_numpy(np.array(y_trn_btch)).float()
    
    val_features = torch.from_numpy(np.array(x_val_btch)).float()
    val_targets = torch.from_numpy(np.array(y_val_btch)).float()
    
    trainval_features = torch.from_numpy(np.array(x_trnval_btch)).float()
    trainval_targets = torch.from_numpy(np.array(y_trnval_btch)).float()
    
    test_features = torch.from_numpy(np.array(x_tst_btch)).float()
    test_targets = torch.from_numpy(np.array(y_tst_btch)).float()
    
    prepped_model_io_data = {'train_features':train_features, 'train_targets':train_targets,
                             'val_features':val_features, 'val_targets': val_targets,
                             'trainval_features':trainval_features, 'trainval_targets':trainval_targets,
                             'test_features':test_features, 'test_targets':test_targets,
                             'means_stds':means_stds}
    
    with open(os.path.join(out_dir,'prepped_model_io_data'), 'wb') as handle:
        pickle.dump(prepped_model_io_data, handle)

def write_model_params(out_dir, run_id, inputs, n_epochs,
                       learn_rate, seq_len, hidden_units,
                       recur_dropout, dropout,
                       train_start_date, train_end_date,
                       val_start_date, val_end_date,
                       test_start_date, test_end_date, inc_ante):
    '''
    Write model parameters to directory where model results will be saved

    Parameters
    ----------
    out_dir : str
        directory where model results will be saved
    run_id : str
        sub-directory within out_dir where model results will be saved
    inputs : list
        list of input variables used as predictors
    n_epochs : int
        number of epochs to run model
    learn_rate : float
        learning rate
    seq_len : int
        sequence length
    hidden_units : int
        number of hidden units in the model
    recur_dropout : float
        fraction of the units to drop from the cell update vector
    dropout : float
        fraction of the units to drop from the input
    train_start_date : str
        start of training in YYYY-MM-DD format
    train_end_date : str
        end of training in YYYY-MM-DD format
    val_start_date : str
        start of validation in YYYY-MM-DD format
    val_end_date : str
        end of validation in YYYY-MM-DD format
    test_start_date : str
        start of testing in YYYY-MM-DD format
    test_end_date : str
        end of testing in YYYY-MM-DD format
    inc_ante : bool
        flag to determine if antecedant conditions should be explicitly included, if True, 
        a rolling look back average of the last n days will be included as an input variable, 
        the number of days and variable are written in the config file

    Returns
    -------
    None.

    '''
    
    
    #write out model paramters
    with open("03b_model/model_config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)
    
    dir = os.path.join(out_dir, run_id)
    os.makedirs(dir, exist_ok=True)
    # if os.path.exists(dir): 
    #     overwrite = input('Directory for this run id already exists, would you like to continue? (yes or no)')
    #     if overwrite == 'yes':
    #         os.makedirs(dir, exist_ok=True)
    #     else:
    #         exit('Nothing written to file')
    # else:
    #     os.mkdir(dir)
    
    inputs_log = inputs.copy()
    
    if inc_ante:
        for var in config['antecedant_variables']: 
            for w in config['window_size']: 
                inputs_log.append(var+'_'+str(w).rjust(3,'0')+'_mean')
    
    f = open(os.path.join(dir,"model_param_output.txt"),"w+")
    f.write("Date: %s\r\n" % date.today().strftime("%b-%d-%Y"))
    f.write("Feature List: %s\r\n" % inputs_log)
    f.write("Target: %s\r\n" % target)
    f.write("Include antecedant variable: %s\r\n" % inc_ante)
    f.write("Epochs: %d\r\n" % n_epochs)
    f.write("Learning rate: %f\r\n" % learn_rate)
    f.write("Sequence Length: %d\r\n" % seq_len)
    f.write("Cells: %d\r\n" % hidden_units)
    f.write("Recurrent Dropout: %f\r\n" % recur_dropout)
    f.write("Dropout: %f\r\n" % dropout)
    f.write("Train date start: %s\r\n" % train_start_date)
    f.write("Train date end: %s\r\n" % train_end_date)
    f.write("Validation date start: %s\r\n" % val_start_date)
    f.write("Validation date end: %s\r\n" % val_end_date)
    f.write("Test date start: %s\r\n" % test_start_date)
    f.write("Test date end: %s\r\n" % test_end_date)
    f.close()

def train_model(prepped_model_io_data_file, inputs, seq_len,
                hidden_units, recur_dropout, 
                dropout, n_epochs, learn_rate, 
                out_dir, run_id,                       
                train_start_date, train_end_date,
                val_start_date, val_end_date,
                test_start_date, test_end_date, inc_ante):
    '''
    write modeling parameters to a .txt file within out_dir/run_id, train the model,
    save the weights, and save a plot of the losses
    Parameters
    ----------
    prepped_model_io_data_file : str
        location of prepped model input, saved as a pickle file by write model params
    inputs : list
        list of input variables used as predictors
    seq_len : int
        sequence length
    hidden_units : int
        number of hidden units in the model
    recur_dropout : float
        fraction of the units to drop from the cell update vector
    dropout : float
        fraction of the units to drop from the input
    n_epochs : int
        number of epochs to run model
    learn_rate : float
        learning rate
    out_dir : str
        directory where model results will be saved
    run_id : str
        sub-directory within out_dir where model results will be saved
    train_start_date : str
        start of training in YYYY-MM-DD format
    train_end_date : str
        end of training in YYYY-MM-DD format
    val_start_date : str
        start of validation in YYYY-MM-DD format
    val_end_date : str
        end of validation in YYYY-MM-DD format
    test_start_date : str
        start of testing in YYYY-MM-DD format
    test_end_date : str
        end of testing in YYYY-MM-DD format
    inc_ante : bool
        flag to determine if antecedant conditions should be explicitly included, if True, 
        a rolling look back average of the last n days will be included as an input variable, 
        the number of days and variable are written in the config file

    Returns
    -------
    None.

    '''
    write_model_params(out_dir, run_id, inputs, n_epochs,
                           learn_rate, seq_len, hidden_units,
                           recur_dropout, dropout,
                           train_start_date, train_end_date,
                           val_start_date, val_end_date,
                           test_start_date, test_end_date, inc_ante)
    
    with open(prepped_model_io_data_file, 'rb') as f:
        prepped_model_io_data = pickle.load(f)
    
    n_batch, seq_len, n_feat  = prepped_model_io_data['train_features'].shape
    pretrain_model = LSTMDA(n_feat, hidden_units, recur_dropout, dropout)
    pretrain_model.train() # ensure that dropout layers are active
    print("fitting model")
    pretrain_model, preds_train, rl_t, rl_v = fit_torch_model(model = pretrain_model, 
                                                              x = prepped_model_io_data['train_features'], 
                                                              y = prepped_model_io_data['train_targets'], 
                                                              x_val = prepped_model_io_data['val_features'], 
                                                              y_val = prepped_model_io_data['val_targets'], 
                                                              epochs = n_epochs, 
                                                              loss_fn = rmse_masked, 
                                                              optimizer = torch.optim.Adam(pretrain_model.parameters(), 
                                                                                           lr = learn_rate))
    
    torch.save(pretrain_model.state_dict(), os.path.join(out_dir, run_id, 'weights.pt'))
    
    plt.plot(rl_t, 'b', label = 'training')
    plt.plot(rl_v,'r', label = 'validation')
    plt.ylabel('Loss (RMSE scaled river mile)')
    plt.legend()
    plt.savefig(os.path.join(out_dir,run_id,'losses.png'))
    plt.close()

def make_predictions(prepped_model_io_data_file, 
                     hidden_units, recur_dropout, dropout, 
                     n_epochs, learn_rate, out_dir, run_id,
                     train_start_date, train_end_date,
                     val_start_date, val_end_date,
                     test_start_date, test_end_date):
    '''
    read weights from file, and make predictions. output results as dataframe   
    
    Parameters
    ----------
    prepped_model_io_data_file : str
        location of prepped model input, saved as a pickle file by write model params
    hidden_units : int
        number of hidden units in the model
    recur_dropout : float
        fraction of the units to drop from the cell update vector
    dropout : float
        fraction of the units to drop from the input
    n_epochs : int
        number of epochs to run model
    learn_rate : float
        learning rate
    out_dir : str
        directory where model results will be saved
    run_id : str
        sub-directory within out_dir where model results will be saved
    train_start_date : str
        start of training in YYYY-MM-DD format
    train_end_date : str
        end of training in YYYY-MM-DD format
    val_start_date : str
        start of validation in YYYY-MM-DD format
    val_end_date : str
        end of validation in YYYY-MM-DD format
    test_start_date : str
        start of testing in YYYY-MM-DD format
    test_end_date : str
        end of testing in YYYY-MM-DD format

    Returns
    -------
    trainval_df : TYPE
        DESCRIPTION.

    '''
    
    
    with open(prepped_model_io_data_file, 'rb') as f:
        prepped_model_io_data = pickle.load(f)
    
    n_batch, seq_len, n_feat  = prepped_model_io_data['train_features'].shape
    
    pretrain_model = LSTMDA(n_feat, hidden_units, recur_dropout, dropout)
    
    pretrain_model.load_state_dict(torch.load(os.path.join(out_dir,run_id,'weights.pt'))) # ensure that dropout layers are active
    
    #for prediction+validation period
    preds_trainval, loss_trainval = pretrain_model.evaluate(x_val = prepped_model_io_data['trainval_features'], y_val = prepped_model_io_data['trainval_targets'])
    
    #get means and standard deviations from input
    means_stds = prepped_model_io_data['means_stds']
    
    #unnormalize
    #predictions for train val set
    preds_trainval_c  = preds_trainval.detach().numpy().reshape(preds_trainval.shape[0]*preds_trainval.shape[1],preds_trainval.shape[2])
    unnorm_trainval = ((preds_trainval_c*means_stds['y_std_trnval']['saltfront_daily'].data)+means_stds['y_mean_trnval']['saltfront_daily'].data).squeeze()
    #known values for trainval set
    known_trainval_c = prepped_model_io_data['trainval_targets'].detach().numpy().reshape(prepped_model_io_data['trainval_targets'].shape[0]*prepped_model_io_data['trainval_targets'].shape[1], prepped_model_io_data['trainval_targets'].shape[2]).squeeze()
    unnorm_known_trainval = (known_trainval_c*means_stds['y_std_trnval']['saltfront_daily'].data)+means_stds['y_mean_trnval']['saltfront_daily'].data
    trainval_dates = pd.date_range(start = train_start_date, periods = known_trainval_c.shape[0], freq = 'D')
    
    
    training_data_length =  prepped_model_io_data['train_targets'].shape[0]*prepped_model_io_data['train_targets'].shape[1]
    
    trainval_df = pd.DataFrame({'saltfront_obs' : unnorm_known_trainval,
                               'saltfront_pred' : unnorm_trainval, 
                               'train/val' : np.repeat(['Training','Validation'], [training_data_length, trainval_dates.shape[0]-training_data_length])},
                               index = trainval_dates)
    
    return trainval_df

def plot_save_predictions(trainval_df, out_dir, run_id):
    '''
    make a plot of the predictions and save to out_dir/run_id
    Parameters
    ----------
    trainval_df : dataframe
        dataframe of training and validation predictions
    out_dir : str
        directory where model results will be saved
    run_id : str
        sub-directory within out_dir where model results will be saved

    Returns
    -------
    None.

    '''
    
    MSE_train = np.square(np.subtract(trainval_df[trainval_df['train/val'] == 'Training']['saltfront_obs'],trainval_df[trainval_df['train/val'] == 'Training']['saltfront_pred'])).mean() 
    RMSE_train = math.sqrt(MSE_train)
    
    MSE_val = np.square(np.subtract(trainval_df[trainval_df['train/val'] == 'Validation']['saltfront_obs'],trainval_df[trainval_df['train/val'] == 'Validation']['saltfront_pred'])).mean() 
    RMSE_val = math.sqrt(MSE_val)
    
    #plot predictions
    plt.plot(trainval_df['saltfront_obs'], 'darkgray', label = 'observed')
    plt.plot(trainval_df['saltfront_pred'], 'darkblue', label = 'training | RMSE: '+str(round(RMSE_train,4)))
    plt.plot(trainval_df[trainval_df['train/val'] == 'Validation']['saltfront_pred'], 'dodgerblue', label = 'validation | RMSE: '+str(round(RMSE_val,4)))
    plt.legend(title = 'Performanace')
    plt.ylabel('River mile')
    #plt.show()
    plt.savefig(os.path.join(out_dir,run_id,'ModelResultsTimeSeries.png'))
    plt.close()
    
    #save predictions
    trainval_df.to_csv(os.path.join(out_dir, run_id, 'ModelResults.csv'))
    
def run_replicates(n_reps, prepped_model_io_data_file):
    '''
    run model replicates to understand variability
    Parameters
    ----------
    n_reps : int
        number of replicates to run
    prepped_model_io_data_file : str
        location of prepped model input, saved as a pickle file by write model params

    Returns
    -------
    None.

    '''
    
    for i in range(config['replicates']):
        
        run_id = os.path.join(config['run_id'], str(i).rjust(2,'0'))
        
        train_model(prepped_model_io_data_file, inputs, seq_len,
                        hidden_units, recur_dropout, 
                        dropout, n_epochs, learn_rate, 
                        out_dir, run_id,                       
                        train_start_date, train_end_date,
                        val_start_date, val_end_date,
                        test_start_date, test_end_date, inc_ante)
        
        predictions = make_predictions(prepped_model_io_data_file, 
                             hidden_units, recur_dropout, dropout, 
                             n_epochs, learn_rate, out_dir, run_id,
                             train_start_date, train_end_date,
                             val_start_date, val_end_date,
                             test_start_date, test_end_date)
        
        plot_save_predictions(predictions, out_dir, run_id)
        

def test_hyperparameters():
    with open("03b_model/hyperparameter_config.yaml", 'r') as stream:
        hp_config = yaml.safe_load(stream)
        
    inputs_xarray, target_xarray = select_inputs_targets(inputs, target, train_start_date, test_end_date, out_dir, inc_ante) 

    hyper_params = hp_config['hyper_params']
    #print('Hyperparameters being tested: '+ hyper_params)
    
    sl = hp_config['seq_len']
    hu = hp_config['hidden_units']
    lr = hp_config['learn_rate']
    do = hp_config['dropout']

    hp_tune_vals = list(itertools.product(sl, hu, lr, do))

    for j in range(len(hp_tune_vals)):
        seq_len = hp_tune_vals[j][0]
        hidden_units = hp_tune_vals[j][1]
        learn_rate = hp_tune_vals[j][2]
        dropout = hp_tune_vals[j][3]
        
        prep_input_target_data(inputs_xarray, target_xarray, train_start_date, train_end_date, 
                               val_start_date, val_end_date, test_start_date, test_end_date, 
                               seq_len, offset, out_dir)
        
        if os.path.exists(os.path.join(out_dir,'prepped_model_io_data')):
           prepped_model_io_data_file = os.path.join(out_dir,'prepped_model_io_data')
        
        for i in range(config['replicates']):
            
            run_id = os.path.join(config['run_id'], str(i).rjust(2,'0'))
            
            train_model(prepped_model_io_data_file, inputs, seq_len,
                            hidden_units, recur_dropout, 
                            dropout, n_epochs, learn_rate, 
                            out_dir, run_id,                       
                            train_start_date, train_end_date,
                            val_start_date, val_end_date,
                            test_start_date, test_end_date, inc_ante)
            
            predictions = make_predictions(prepped_model_io_data_file, 
                                 hidden_units, recur_dropout, dropout, 
                                 n_epochs, learn_rate, out_dir, run_id,
                                 train_start_date, train_end_date,
                                 val_start_date, val_end_date,
                                 test_start_date, test_end_date)
            
            plot_save_predictions(predictions, out_dir, run_id)


# A function for training the model on weighted training data
# def train_high():
#     out_dir = os.path.join(config['out_dir'],config['run_id'],'train_high')
#     inputs_xarray_high, target_xarray_high = select_inputs_targets(inputs, target, train_start_date, test_end_date, out_dir, inc_ante, train_high = True) 
      
#     prep_input_target_data(inputs_xarray_high, target_xarray_high, train_start_date, train_end_date, 
#                            val_start_date, val_end_date, test_start_date, test_end_date, 
#                            seq_len, offset, out_dir)
    
#     prepped_model_io_data_file = os.path.join(out_dir,'prepped_model_io_data')
    
#     train_model(prepped_model_io_data_file, inputs, seq_len,
#                     hidden_units, recur_dropout, 
#                     dropout, n_epochs, learn_rate, 
#                     out_dir, run_id,                       
#                     train_start_date, train_end_date,
#                     val_start_date, val_end_date,
#                     test_start_date, test_end_date, inc_ante)
    
#     out_dir = os.path.join(config['out_dir'],config['run_id'],'predict all')
#     inputs_xarray, target_xarray = select_inputs_targets(inputs, target, train_start_date, test_end_date, out_dir, inc_ante, train_high = False) 
    
#     prep_input_target_data(inputs_xarray_high, target_xarray_high, train_start_date, train_end_date, 
#                            val_start_date, val_end_date, test_start_date, test_end_date, 
#                            seq_len, offset, out_dir)
    
#     prepped_model_io_data_file = os.path.join(out_dir,'prepped_model_io_data')
#     predictions = make_predictions(prepped_model_io_data_file, 
#                          hidden_units, recur_dropout, dropout, 
#                          n_epochs, learn_rate, out_dir, run_id,
#                          train_start_date, train_end_date,
#                          val_start_date, val_end_date,
#                          test_start_date, test_end_date)
    
#     plot_save_predictions(predictions, out_dir, run_id)
    
    

def main():
    '''
    prepare inputs and targets
    train the model, save the weights
    make predictions
    save and plot predictions

    Returns
    -------
    None.

    '''
        
    inputs_xarray, target_xarray = select_inputs_targets(inputs, target, train_start_date, test_end_date, out_dir, inc_ante) 
    
    
    prep_input_target_data(inputs_xarray, target_xarray, train_start_date, train_end_date, 
                           val_start_date, val_end_date, test_start_date, test_end_date, 
                           seq_len, offset, out_dir)
    
    if os.path.exists(os.path.join(out_dir,'prepped_model_io_data')):
       prepped_model_io_data_file = os.path.join(out_dir,'prepped_model_io_data')
    
    
    train_model(prepped_model_io_data_file, inputs, seq_len,
                    hidden_units, recur_dropout, 
                    dropout, n_epochs, learn_rate, 
                    out_dir, run_id,                       
                    train_start_date, train_end_date,
                    val_start_date, val_end_date,
                    test_start_date, test_end_date, inc_ante)
    
    predictions = make_predictions(prepped_model_io_data_file, 
                         hidden_units, recur_dropout, dropout, 
                         n_epochs, learn_rate, out_dir, run_id,
                         train_start_date, train_end_date,
                         val_start_date, val_end_date,
                         test_start_date, test_end_date)
    
    plot_save_predictions(predictions, out_dir, run_id)
    

if __name__ == '__main__':
    main()
    

