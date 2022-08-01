# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 15:44:55 2022

@author: ggorski
"""

from datetime import date
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

###
def select_inputs_targets(inputs, target, train_start_date, test_end_date, out_dir):
    '''select the variables you are interested in examining, 
    srcs must be a list using the exact variable names,
    it will return a list of dataframes, each dataframe corresponding to a site
    with the requested variables as columns'''
    inputs_df = pd.DataFrame()
    
    #print('Looking for sources: ', inputs)
    
    files_to_read = []
    for top,dirs,files in os.walk('02_munge/out/'):
        if top == '02_munge/out/D' or top == '02_munge/out/daily_summaries':
            files_to_read.extend([os.path.join(top,files) for files in files])
    
    for inp in inputs:
        var = '_'.join(inp.split('_')[:-1])
        site = inp.split('_')[-1]
        
        #if site == 'delsjmet':
        #    continue
        
        inp_file = [s for s in files_to_read if site in s]
        data = pd.read_csv(inp_file[0], parse_dates = True, index_col = 'datetime')
        try:
            inputs_df[inp] = data[var]
        except:
            continue
    
    #inputs_df = inputs_df.join(nerrs_input)
    inputs_df = inputs_df[train_start_date:test_end_date]
    
    
    #read in the salt front record
    target_df = pd.read_csv(os.path.join('03a_it_analysis', 'in', 'saltfront.csv'), parse_dates = True, index_col = 'datetime')
    target_df = target_df['saltfront_daily'].to_frame()
    target_df.index = pd.to_datetime(target_df.index.date)
    target_df = target_df[train_start_date:test_end_date]
    target_df.index = target_df.index.rename('datetime')
    
    #set everything below 54 to nan
    target_df_c = target_df.copy()
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
                           seq_len, offset, out_dir, return_data):
    
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
    
    x_scl_val, x_std_val, x_mean_val = scale(x_val)
    y_scl_val, y_std_val, y_mean_val = scale(y_val)
    
    x_scl_trnval, x_std_trnval, x_mean_trnval = scale(x_trnval)
    y_scl_trnval, y_std_trnval, y_mean_trnval = scale(y_trnval)
    
    x_scl_tst, x_std_tst, x_mean_tst = scale(x_tst)
    y_scl_tst, y_std_tst, y_mean_tst = scale(y_tst)
    
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
    if return_data:
        return prepped_model_io_data
    
    with open(os.path.join(out_dir,'prepped_model_io_data'), 'wb') as handle:
        pickle.dump(prepped_model_io_data, handle)

def write_model_params(out_dir, run_id, inputs, n_epochs_pre,
                       learn_rate_pre, seq_len, hidden_units,
                       train_start_date, train_end_date,
                       val_start_date, val_end_date,
                       test_start_date, test_end_date):
    #write out model paramters
    dir = os.path.join(out_dir, run_id)
    if os.path.exists(dir): 
        overwrite = input('Directory for this run id already exists, would you like to continue? (yes or no)')
        if overwrite == 'yes':
            os.makedirs(dir, exist_ok=True)
        else:
            exit('Nothing written to file')
    else:
        os.mkdir(dir)
    
    f= open(os.path.join(dir,"model_param_output.txt"),"w+")
    f.write("Date: %s\r\n" % date.today().strftime("%b-%d-%Y"))
    f.write("Feature List: %s\r\n" % inputs)
    f.write("Epochs: %d\r\n" % n_epochs_pre)
    f.write("Learning rate: %f\r\n" % learn_rate_pre)
    f.write("Sequence Length: %d\r\n" % seq_len)
    f.write("Cells: %d\r\n" % hidden_units)
    f.write("Train date start: %s\r\n" % train_start_date)
    f.write("Train date end: %s\r\n" % train_end_date)
    f.write("Validation date start: %s\r\n" % val_start_date)
    f.write("Validation date end: %s\r\n" % val_end_date)
    f.write("Test date start: %s\r\n" % test_start_date)
    f.write("Test date start: %s\r\n" % test_end_date)
    f.close()

def train_model(prepped_model_io_data_file, inputs, seq_len,
                hidden_units, recur_dropout, 
                dropout, n_epochs_pre, learn_rate_pre, 
                out_dir, run_id,                       
                train_start_date, train_end_date,
                val_start_date, val_end_date,
                test_start_date, test_end_date):
    
    write_model_params(out_dir, run_id, inputs, n_epochs_pre,
                           learn_rate_pre, seq_len, hidden_units,
                           train_start_date, train_end_date,
                           val_start_date, val_end_date,
                           test_start_date, test_end_date)
    
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
                                                              epochs = n_epochs_pre, 
                                                              loss_fn = rmse_masked, 
                                                              optimizer = torch.optim.Adam(pretrain_model.parameters(), 
                                                                                           lr = learn_rate_pre))
    
    torch.save(pretrain_model.state_dict(), os.path.join(out_dir, run_id, 'weights.pt'))
    
    plt.plot(rl_t, 'b', label = 'training')
    plt.plot(rl_v,'r', label = 'validation')
    plt.ylabel('RMSE river mile')
    plt.legend()
    #plt.show()
    plt.savefig(os.path.join(out_dir,run_id,'losses.png'))
    plt.close()

def make_predictions(prepped_model_io_data_file, 
                     hidden_units, recur_dropout, dropout, 
                     n_epochs_pre, learn_rate_pre, out_dir, run_id,
                     train_start_date, train_end_date,
                     val_start_date, val_end_date,
                     test_start_date, test_end_date):
    
    
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


def main():
    with open("03b_model/model_config.yaml", 'r') as stream:
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
    
    inputs_xarray, target_xarray = select_inputs_targets(inputs, target, train_start_date, test_end_date, out_dir) 
    
    prep_input_target_data(inputs_xarray, target_xarray, train_start_date, train_end_date, 
                           val_start_date, val_end_date, test_start_date, test_end_date, 
                           seq_len, offset, out_dir)
    
    if os.path.exists(os.path.join(out_dir,'prepped_model_io_data')):
       prepped_model_io_data_file = os.path.join(out_dir,'prepped_model_io_data')
    
      
    train_model(prepped_model_io_data_file, inputs, seq_len,
                    hidden_units, recur_dropout, 
                    dropout, n_epochs_pre, learn_rate_pre, 
                    out_dir, run_id,                       
                    train_start_date, train_end_date,
                    val_start_date, val_end_date,
                    test_start_date, test_end_date)
    
    predictions = make_predictions(prepped_model_io_data_file, 
                         hidden_units, recur_dropout, dropout, 
                         n_epochs_pre, learn_rate_pre, out_dir, run_id,
                         train_start_date, train_end_date,
                         val_start_date, val_end_date,
                         test_start_date, test_end_date)
    
    plot_save_predictions(predictions, out_dir, run_id)
    

if __name__ == '__main__':
    main()