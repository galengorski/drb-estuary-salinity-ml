# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 10:17:34 2022

@author: ggorski
"""
import it_functions
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats.stats import pearsonr 


#%%
# ndata = 365

# n_lags = 25

# #choatic logistic mapping
# #growth rate
# a = 4
# #coupling lag
# lag = 5
# #noise (0,1)
# e = 0.5
# #type of link 
# map_type = 'logistic'

# #
# cca = 1
# cc2 = 5
# cc3 = 1
# cc4 = 4
def generate_logistic_data(ndata, growth, lag, e):
    x = np.random.rand(ndata)
    y = np.zeros(ndata)
    z = np.random.randn(ndata)

    for i in np.arange(lag,ndata):
        y[i] = growth*x[i-lag]*(1-x[i-lag])+e*z[i]
    
    # Normalize 
    x = (x - np.min(x))/(np.max(x) - np.min(x))
    y = (y - np.min(y))/(np.max(y) - np.min(y))
     
    M = np.stack([x,y], axis = 1)
    M = M[np.s_[lag:]]
     
    Mswap = np.stack([y,x], axis = 1)
    Mswap = Mswap[np.s_[lag:]]
     
    return M, Mswap

def generate_periodic_data(ndata, cc1, cc2, cc3, cc4, e):
    x = np.random.rand(ndata)
    y = np.zeros(ndata)
    z = np.random.randn(ndata)

    for i in np.arange(4,ndata):
        y[i] = cc1*np.exp(x[i-1]) + cc2*x[i-2]**2 + cc3*x[i-3] + cc4*np.cos(x[i-4]) + e*z[i]
 
    # Normalize 
    x = (x - np.min(x))/(np.max(x) - np.min(x))
    y = (y - np.min(y))/(np.max(y) - np.min(y))
    
    M = np.stack([x,y], axis = 1)
    M = M[np.s_[4:]]
    
    Mswap = np.stack([y,x], axis = 1)
    Mswap = Mswap[np.s_[4:]]
    
    return M, Mswap

def generate_independent_data(ndata):
    x = np.random.rand(ndata)
    y = np.random.rand(ndata)
        
    # Normalize |
    x = (x - np.min(x))/(np.max(x) - np.min(x))
    y = (y - np.min(y))/(np.max(y) - np.min(y))
    
    M = np.stack([x,y], axis = 1)
    
    Mswap = np.stack([y,x], axis = 1)
    
    return M, Mswap


def generate_data(ndata, map_type, log_growth, log_lag, cc1, cc2, cc3, cc4, e):
    
    if map_type == 'logistic':
        lag = log_lag
    else:
        lag = 4
    
    x = np.random.rand(ndata)
    y = np.zeros(ndata)
    z = np.random.randn(ndata)

    for i in np.arange(lag,ndata):
        if map_type == 'logistic':
            y[i] = log_growth*x[i-lag]*(1-x[i-lag])+e*z[i]
        elif map_type == 'periodic':
            y[i] = cc1*np.exp(x[i-1]) + cc2*x[i-2]**2 + cc3*x[i-3] + cc4*np.cos(x[i-4]) + e*z[i]
        elif map_type == 'independent':
            y[i] = np.random.rand(1)
    # Normalize |
    x = (x - np.min(x))/(np.max(x) - np.min(x))
    y = (y - np.min(y))/(np.max(y) - np.min(y))
    
    M = np.stack([x,y], axis = 1)
    M = M[np.s_[lag:]]
    
    Mswap = np.stack([y,x], axis = 1)
    Mswap = Mswap[np.s_[lag:]]
    
    return M, Mswap
#%%
#plt.plot(M[:,0], label = 'Source x')
#plt.plot(M[:,1], label = 'Sink y')
#plt.legend()
#%%
# plt.hist(M[:,0])
# plt.hist(M[:,1])
# plt.ylabel('Frequency')
# plt.xlabel('Value')
# plt.legend(('Source x','Sink y' ),loc='upper right')

#%%

def calc_it_metrics(M, Mswap, n_lags, calc_swap = True, nbins = 11, alpha = 0.01):
    MI = []
    MIcrit = []
    corr = []
    TE = []
    TEcrit = []
    TEswap = []
    TEcritswap = []
    for i in range(1,n_lags):
        #lag data
        M_lagged = it_functions.lag_data(M,shift = i)
        #remove any rows where there is an nan value
        M_short =  M_lagged[~np.isnan(M_lagged).any(axis=1)]
        MItemp = it_functions.calcMI(M_short[:,(0,1)], nbins = nbins)
        MI.append(MItemp)
        MIcrittemp = it_functions.calcMI_crit(M_short[:,(0,1)], nbins = nbins, ncores = 8, alpha = 0.01)
        MIcrit.append(MIcrittemp)
        
        corrtemp = pearsonr(M_short[:,0], M_short[:,1])[0]
        corr.append(corrtemp)
        
        TEtemp = it_functions.calcTE(M, shift = i, nbins = nbins)
        TE.append(TEtemp)
        TEcrittemp = it_functions.calcTE_crit(M, shift = i, nbins = nbins, ncores = 8, alpha = 0.01)
        TEcrit.append(TEcrittemp)
        
        if calc_swap:
            TEtempswap = it_functions.calcTE(Mswap, shift = i)
            TEswap.append(TEtempswap)
            TEcrittempswap = it_functions.calcTE_crit(Mswap, shift = i, nbins = nbins, ncores = 8, alpha = 0.01)
            TEcritswap.append(TEcrittempswap)
        
    it_metrics = {'MI':MI, 'MIcrit':MIcrit,
                  'TE':TE, 'TEcrit':TEcrit,
                  'TEswap':TEswap, 'TEcritswap':TEcritswap,
                  'corr':corr}
    
    return it_metrics
    
#%%
def plot_te(n_lags, it_metrics, plot_swap = True):
    plt.plot(range(1,n_lags),it_metrics['TE'], color='#0077b6', marker='o', linewidth=2, markersize=5, label = 'TE x -> y')
    plt.plot(range(1,n_lags),it_metrics['TEcrit'], color = '#00b4d8', linewidth=1, linestyle='dashed', label = 'TE critical x -> y')
    if plot_swap:
        plt.plot(range(1,n_lags),it_metrics['TEswap'], color='#ff5400', marker='o', linewidth=2, markersize=5, label = 'TE y -> x')
        plt.plot(range(1,n_lags),it_metrics['TEcritswap'], color = '#ff9e00', linewidth=1, linestyle='dashed', label = 'TE critical y -> x')
    plt.xlabel('Time lag')
    plt.ylabel('TE as a fraction of H(y)')
    plt.title('Transfer entropy')
    plt.legend()
    plt.show()
#%%
def plot_mi_corr(n_lags, it_metrics):
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    mi_p = ax1.plot(range(1,n_lags),it_metrics['MI'], color='dodgerblue', marker='o', linewidth=2, markersize=5, label = 'MI')
    micrit_p = ax1.plot(range(1,n_lags),it_metrics['MIcrit'], color = 'lightskyblue', linewidth=2, linestyle='dashed', label = 'MI critical')
    corr_p = ax2.plot(range(1,n_lags),it_metrics['corr'], color='red', marker='o', linewidth=2, markersize=5, label = 'Pearson correlation')
    ax1.set_xlabel('Time lag')
    ax1.set_ylabel('MI as a fraction of H(y)', color='dodgerblue')
    ax2.set_ylabel('Pearson correlation', color='red')
    # Solution for having two legends
    leg = mi_p +micrit_p + corr_p
    labs = [l.get_label() for l in leg]
    ax1.legend(leg, labs, loc=0)
    plt.show()

#%%

def gen_plot_it_metrics(ndata, map_type, log_growth, log_lag, cc1, cc2, cc3, cc4, e, n_lags_plot):
    M, Mswap = generate_data(ndata, map_type, log_growth, log_lag, cc1, cc2, cc3, cc4, e)
    it_metrics = calc_it_metrics(M,Mswap,n_lags_plot)
    plot_te(n_lags_plot, it_metrics)
    plot_mi_corr(n_lags_plot, it_metrics)
    
#%%
def gen_plot_logistic_it_te(ndata, e): 
    growth = 4
    lag = 5
    n_lags_plot = 10
    calc_swap = True
    M, Mswap = generate_logistic_data(ndata, growth, lag, e)
    it_dict = calc_it_metrics(M,Mswap, n_lags_plot ,calc_swap)
    plt.plot(range(1,n_lags_plot),it_dict['TE'], color='#0077b6', marker='o', linewidth=2, markersize=5, label = 'TE x -> y')
    plt.plot(range(1,n_lags_plot),it_dict['TEcrit'], color = '#00b4d8', linewidth=1, linestyle='dashed', label = 'TE critical x -> y')
    if calc_swap:
        plt.plot(range(1,n_lags_plot),it_dict['TEswap'], color='#ff5400', marker='o', linewidth=2, markersize=5, label = 'TE y -> x')
        plt.plot(range(1,n_lags_plot),it_dict['TEcritswap'], color = '#ff9e00', linewidth=1, linestyle='dashed', label = 'TE critical y -> x')
    plt.xlabel('Time lag')
    plt.ylabel('TE as a fraction of H(y)')
    plt.title('Transfer entropy | number of data points = '+str(ndata) + ' | noise = '+ str(e))
    plt.legend()
    plt.show()
#%%
def gen_plot_logistic_it_mi(ndata, e): 
    growth = 4
    lag = 5
    n_lags_plot = 10
    calc_swap = False
    M, Mswap = generate_logistic_data(ndata, growth, lag, e)
    it_dict = calc_it_metrics(M,Mswap, n_lags_plot ,calc_swap)
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    mi_p = ax1.plot(range(1,n_lags_plot),it_dict['MI'], color='dodgerblue', marker='o', linewidth=2, markersize=5, label = 'MI')
    micrit_p = ax1.plot(range(1,n_lags_plot),it_dict['MIcrit'], color = 'lightskyblue', linewidth=2, linestyle='dashed', label = 'MI critical')
    corr_p = ax2.plot(range(1,n_lags_plot),it_dict['corr'], color='red', marker='o', linewidth=2, markersize=5, label = 'Pearson correlation')
    ax1.set_xlabel('Time lag')
    ax1.set_ylabel('MI as a fraction of H(y)', color='dodgerblue')
    ax2.set_ylabel('Pearson correlation', color='red')
    fig.suptitle('Transfer entropy | number of data points = '+str(ndata) + ' | noise = '+ str(e))
    # Solution for having two legends
    leg = mi_p +micrit_p + corr_p
    labs = [l.get_label() for l in leg]
    ax1.legend(leg, labs, loc=0)
    plt.show()
#%%
def gen_plot_periodic_it_te(ndata, cc1, cc2, cc3, cc4, e): 
    n_lags_plot = 10
    calc_swap = True
    M, Mswap = generate_periodic_data(ndata, cc1, cc2, cc3, cc4, e)
    it_dict = calc_it_metrics(M,Mswap, n_lags_plot ,calc_swap)
    plt.plot(range(1,n_lags_plot),it_dict['TE'], color='#0077b6', marker='o', linewidth=2, markersize=5, label = 'TE x -> y')
    plt.plot(range(1,n_lags_plot),it_dict['TEcrit'], color = '#00b4d8', linewidth=1, linestyle='dashed', label = 'TE critical x -> y')
    if calc_swap:
        plt.plot(range(1,n_lags_plot),it_dict['TEswap'], color='#ff5400', marker='o', linewidth=2, markersize=5, label = 'TE y -> x')
        plt.plot(range(1,n_lags_plot),it_dict['TEcritswap'], color = '#ff9e00', linewidth=1, linestyle='dashed', label = 'TE critical y -> x')
    plt.xlabel('Time lag')
    plt.ylabel('TE as a fraction of H(y)')
    plt.title('Transfer entropy | number of data points = '+str(ndata) + ' | noise = '+ str(e))
    plt.legend()
    plt.show()

#%%
def gen_plot_periodic_it_mi(ndata, cc1, cc2, cc3, cc4, e): 
    n_lags_plot = 10
    calc_swap = False
    M, Mswap = generate_periodic_data(ndata, cc1, cc2, cc3, cc4, e)
    it_dict = calc_it_metrics(M,Mswap, n_lags_plot ,calc_swap)
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    mi_p = ax1.plot(range(1,n_lags_plot),it_dict['MI'], color='dodgerblue', marker='o', linewidth=2, markersize=5, label = 'MI')
    micrit_p = ax1.plot(range(1,n_lags_plot),it_dict['MIcrit'], color = 'lightskyblue', linewidth=2, linestyle='dashed', label = 'MI critical')
    corr_p = ax2.plot(range(1,n_lags_plot),it_dict['corr'], color='red', marker='o', linewidth=2, markersize=5, label = 'Pearson correlation')
    ax1.set_xlabel('Time lag')
    ax1.set_ylabel('MI as a fraction of H(y)', color='dodgerblue')
    ax2.set_ylabel('Pearson correlation', color='red')
    fig.suptitle('Transfer entropy | number of data points = '+str(ndata) + ' | noise = '+ str(e))
    # Solution for having two legends
    leg = mi_p +micrit_p + corr_p
    labs = [l.get_label() for l in leg]
    ax1.legend(leg, labs, loc=0)
    plt.show()
