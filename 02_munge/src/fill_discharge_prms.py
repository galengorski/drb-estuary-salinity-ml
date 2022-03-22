# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 08:01:27 2022

@author: ggorski
"""

import os
import pandas as pd
import sciencebasepy
import yaml
import zipfile




def download_unzip_sb(sb_url, prms_predictions, destination):
    '''downloads prms predictions of discharge from science base data release: https://www.sciencebase.gov/catalog/item/5f6a289982ce38aaa2449135
    :sb_url: [str] url for downloading zip file
    :prms_predictions: [str] specific file name 
    :destination: [str] where the file will be downloaded to'''
    
    os.makedirs(destination, exist_ok=True)
    sb = sciencebasepy.SbSession()
    sb.download_file(sb_url, prms_predictions+'.zip', destination)
    with zipfile.ZipFile(os.path.join(destination,prms_predictions+'.zip'), 'r') as zip_ref:
        zip_ref.extractall(destination)
        


def fill_discharge_prms(trenton_seg_id_nat, schuylkill_seg_id_nat, nwis_trenton_location, nwis_schuylkill_location, destination, prms_predictions):
    '''reads in prms predictions of discharge and fills nan gaps for trenton and schuylkill nwis time series
    :trenton_seg_id_nat: [int] segment id closest to trenton discharge location
    :schuylkill_seg_id_nat: [int] segment id closest to schylkill discharge location
    :nwis_trenton_location: [str] location of munged trenton nwis .csv
    :nwis_schuylkill_location: [str] location of munged schuylkill nwis .csv
    :destination: [str] directory where prms data file is located
    :prms_predictions: [str] name of prms file'''
    
    sn_temp_data = pd.read_csv(os.path.join(destination,prms_predictions+'.csv'), parse_dates = True, index_col = 'date')
    
    sn_temp_trenton = sn_temp_data[sn_temp_data['seg_id_nat'] == trenton_seg_id_nat]
    sn_temp_schuylkill = sn_temp_data[sn_temp_data['seg_id_nat'] == schuylkill_seg_id_nat]
    
    nwis_trenton = pd.read_csv(nwis_trenton_location, parse_dates = True, index_col = 'datetime')
    nwis_schuylkill = pd.read_csv(nwis_schuylkill_location, parse_dates = True, index_col = 'datetime')
    
    #fill the gaps make sure to from cubic meters to cfs
    nwis_trenton['filled'] = nwis_trenton['discharge'].fillna(sn_temp_trenton['seg_outflow']*35.315)
    nwis_schuylkill['filled'] = nwis_schuylkill['discharge'].fillna(sn_temp_schuylkill['seg_outflow']*35.315)
    
    #drop the old discharge and rename filled as discharge
    nwis_trenton = nwis_trenton.drop(columns = 'discharge')
    nwis_trenton['discharge'] = nwis_trenton['filled'].rename('discharge')
    
    nwis_schuylkill = nwis_schuylkill.drop(columns = 'discharge')
    nwis_schuylkill['discharge'] = nwis_schuylkill['filled'].rename('discharge')
    
    #write to dataframe
    nwis_trenton.to_csv(nwis_trenton_location)
    nwis_schuylkill.to_csv(nwis_schuylkill_location)


def main():
    # import config
    with open("02_munge/munge_config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)['fill_discharge_prms.py']
    
    sb_url = config['sb_url']
    prms_predictions = config['prms_predictions']
    destination = config['destination']
    trenton_seg_id_nat = config['trenton_seg_id_nat']
    schuylkill_seg_id_nat = config['schuylkill_seg_id_nat']
    nwis_trenton_location = config['nwis_trenton_location']
    nwis_schuylkill_location = config['nwis_schuylkill_location']
    
    download_unzip_sb(sb_url, prms_predictions, destination)
    fill_discharge_prms(trenton_seg_id_nat, schuylkill_seg_id_nat, nwis_trenton_location, nwis_schuylkill_location, destination, prms_predictions)

if __name__ == '__main__':
    main()
