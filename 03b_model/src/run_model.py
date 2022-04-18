# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 15:44:55 2022

@author: ggorski
"""

import numpy as np
import pandas as pd
import os
import pickle
import yaml
import utils
from it_analysis_data_prep import select_sources, select_sinks


input_data_x = srcs_list['correlation'][0]
for i in range(1,len(srcs_list['correlation'])):
    input_data_x = input_data_x.join(srcs_list['correlation'][i], how = 'outer')
input_data_x = input_data_x.loc[:,~input_data_x.columns.str.contains('lag')]














def main():
    #import config
    with open("03b_model/run_model_config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)['data_prep.py']
    #select all possible sources from munge
    srcs = config['all_srcs']
    #select sinks (target variable)
    snks = config['snks']
    date_start = config['train_start_date']
    #end date for analysis
    date_end = config['test_end_date']

    #generate the sources data
    srcs_list, _ = select_sources(srcs, date_start, date_end)    
    #generate the sinks data
    snks_list, _ = select_sinks(snks, date_start, date_end)

        
        
if __name__ == '__main__':
    main()