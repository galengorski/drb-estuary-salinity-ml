# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 10:11:52 2022

@author: ggorski

These functions were developed borrowing code and ideas from:
    Laurel Larsen (laurel@berkeley.edu) for Larsen, L. G. and J. W. Harvey. 2017. Disrupted carbon cycling in restored and unrestored urban streams: Critical timescales and controls. Limnology and Oceanography, 62(Suppl. S1), S160-S182, doi: 10.1002/lno.10613. Please cite this work accordingly in products that use this code.
    Edom Modges (edom.moges@berkeley.edu)
    Julianne Quinn University of Virginia
    The transfer entropy routine is based on Ruddell and Kumar, 2009, "Ecohydrologic process networks: 1. Identification," Water Resources Research, 45, doi:10.1029/2008WR007279.

"""

from joblib import Parallel, delayed
import numpy as np


def calc2Dpdf(M,nbins = 11):
    '''calculates the 3 pdfs, one for x, one for y and a joint pdf for x and y 
    M: a numpy array of shape (nobs, 2) where nobs is the number of observations
    this assumes that the data are arrange such that the first column is the source (x) and
    the second column is the sink (y).
    nbins: is the number of bins used for estimating the pdf with a default of 11'''
    
    counts, binEdges = np.histogramdd(M,bins=nbins)
    p_xy = counts/np.sum(counts)
    
    p_x = np.sum(p_xy,axis=1)
    p_y = np.sum(p_xy,axis=0)
    
    return p_x, p_y, p_xy



def calcEntropy(pdf):
    '''calculate the entropy from the pdf
    here n_0 is used to indicate that all values are non-zero'''
    
    pdf_n_0 = pdf[pdf>0]
    log2_pdf_n_0 = np.log2(pdf_n_0)
    H = (-sum(pdf_n_0*log2_pdf_n_0))
    return H


def calcMI(M, nbins = 11):
    '''calculate mutual information of two variables
    M: a numpy array of shape (nobs, 2) where nobs is the number of observations
    this assumes that the data are arrange such that the first column is the source and
    the second column is the sink.
    nbins: is the number of bins used for estimating the pdf with a default of 11
    the mutual information is normalized by the entropy of the sink'''
    
    
    p_x, p_y, p_xy = calc2Dpdf(M, nbins = 11)
    
    Hx = calcEntropy(p_x)
    Hy = calcEntropy(p_y)
    Hxy = calcEntropy(p_xy)
    
    MI = (Hx+Hy-Hxy)/Hy
    
    return MI

def calcMI_shuffled(M, nbins = 11):
    '''shuffles the input dataset to destroy temporal relationships for signficance testing
    M: a numpy array of shape (nobs, 2) where nobs is the number of observations
    this assumes that the data are arrange such that the first column is the source and
    the second column is the sink.
    nbins: is the number of bins used for estimating the pdf with a default of 11
    the mutual information is normalized by the entropy of the sink
    returns a single MI value for a numpy array the same size and shape as M, 
    but with the order shuffled'''
    
    Mss = np.ones(np.shape(M))*np.nan # Initialize
    
    for n in range(np.shape(M)[1]): # Columns are shuffled separately
        n_nans = np.argwhere(~np.isnan(M[:,n]))
        R = np.random.rand(np.shape(n_nans)[0],1)
        I = np.argsort(R,axis=0)
        Mss[n_nans[:,0],n] = M[n_nans[I[:],0],n].reshape(np.shape(M[n_nans[I[:],0],n])[0],)
    MI_shuff = calcMI(Mss, nbins = 11)
    return MI_shuff
    
def calcMI_crit(M, nbins = 11, alpha = 0.05, numiter = 1000, ncores = 2):
    '''calculate the critical threshold of mutual information
    M: a numpy array of shape (nobs, 2) where nobs is the number of observations
    this assumes that the data are arrange such that the first column is the source and
    the second column is the sink.
    nbins: is the number of bins used for estimating the pdf with a default of 11
    the mutual information is normalized by the entropy of the sink
    alpha: sigficance threshold, default = 0.05
    numiter: number of iterations, default = 1000
    ncores: number of cores, default = 2'''
    
    MIss = Parallel(n_jobs=ncores)(delayed(calcMI_shuffled)(M, nbins) for ii in range(numiter))
    MIss = np.sort(MIss)
    #print(MIss)
    MIcrit = MIss[round((1-alpha)*numiter)] 
    return(MIcrit)

