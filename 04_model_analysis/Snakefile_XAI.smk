import pickle
import os
import sys
sys.path.insert(0, 'C:\\Users\\ggorski\\OneDrive - DOI\\USGS_ML\\Estuary Salinity\\github\\drb-estuary-salinity-ml\\04_model_analysis\\src')
import XAI_functions

rule expected_gradients_no_temporal:
    input:
        '03b_model/model_config.yaml'
    output:
        '04_model_analysis/out/expected_gradients_w_variables.csv'
    run:
        XAI_functions.calc_EGs_for_model(config_loc = input[0], n_samples = 200, write_loc = output[0])

rule save_single_year_plot:
    input:
        '04_model_analysis/out/expected_gradients_w_variables.csv'
    output:
        '04_model_analysis/out/fig/expected_gradients_mean_reps'
    run:
        XAI_functions.plot_EGs_single_year(eg_df_loc = input[0], 
                                            year = '2019', 
                                            save_loc = output[0],
                                            ext = 'jpg')

rule expected_gradients_all_doy:
    input:
        '03b_model/model_config.yaml'
    output:
        '04_analysis/out/EGs_all_days_df'
    run:
        XAI_functions.calc_Egs_for_all_days(config_loc = input[0], 
                                rep = 0, 
                                n_samples = 200, 
                                write_loc = output[0])

rule plot_eg_lookback:
    input:
        #'04_model_analysis/out/EGs_all_days_df'
    output:
        '04_model_analysis/out/fig/expected_gradient_lookback.jpg'
    run:
        XAI_functions.plot_EGs_lookback(eg_all_days_loc = '04_model_analysis/out/EGs_all_days_df', 
                                        t = 0.99, 
                                        save_loc = output[0])
                                        
rule permutation_feature_importance:
    input:
       '03b_model/model_config.yaml'
    output:
        '04_model_analysis/out/feature_importance.csv'
    run:
        XAI_functions.calc_feature_importance_reps(config_loc = input[0], 
                                                    reps = ['00','01','02','03','04'], 
                                                    write_loc = output[0]) 

rule plot_feature_importance:
    input:
        '04_model_analysis/out/feature_importance.csv'
    output:
        '04_model_analysis/out/fig/permutation_feature_importance.jpg'
    run:
        XAI_functions.plot_feature_importance(feat_imp_loc = input[0], save_loc = output[0])