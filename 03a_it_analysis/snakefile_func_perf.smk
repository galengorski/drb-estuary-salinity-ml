import pickle
import os
import sys
sys.path.insert(0, '03a_it_analysis\\src')
sys.path.append('methods_exploration')
import estuary_salinity_functional_performance_func as esfp

run_id = 'Run_51'
sources = ['discharge_01474500','discharge_01463500']
sinks = ['saltfront_obs', 'ml_pred','coawst_pred']
years = ['2001','2002','2003','2004','2005','2006','2007','2008','2009','2010',
        '2011','2012','2013','2014','2015', '2016','2017', '2018','2019','2020'] 


rule create_cleaned_io_file:
    output:
        os.path.join('03b_model/out',run_id,'ML_COAWST_Results_Input_Cleaned.csv')
    run:
        esfp.create_cleaned_io_file(run_id)

rule calc_it_metrics_for_pairs:
    input:
        os.path.join('03b_model/out',run_id,'ML_COAWST_Results_Input_Cleaned.csv')
    output:
        os.path.join('03a_it_analysis/out',run_id+'_it_df.csv')
    run:
        esfp.calc_it_metrics_for_pairs(data_loc = input[0], 
                                    sources = sources, 
                                    sinks = sinks, 
                                    years = years, 
                                    write_loc = output[0])
                                    
rule calc_functional_performance:
    input:
        os.path.join('03a_it_analysis/out/',run_id+'_it_df.csv')
    output:
        os.path.join('03a_it_analysis/out', run_id+'_functional_performance_df.csv')
    run:
        esfp.calc_functional_performance(it_df_loc = input[0], 
                                        sources = sources, 
                                        sinks = sinks, 
                                        years = years, 
                                        write_loc = output[0])