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

# import config
with open("02_munge/params_config_fill_discharge_prms.yaml", 'r') as stream:
    config = yaml.safe_load(stream)

def download_unzip_sb(sb_url, prms_predictions, destination):
    '''downloads prms predictions of discharge from science base data release: https://www.sciencebase.gov/catalog/item/5f6a289982ce38aaa2449135
    :sb_url: [str] url for downloading zip file
    :prms_predictions: [str] specific file name 
    :destination: [str] where the file will be downloaded to'''
    # make input directory for data if it doesn't exist
    os.makedirs(destination, exist_ok=True)
    if os.path.exists(os.path.join(destination,prms_predictions+'.zip')):
        print('File has already been downloaded')
    else:
        os.makedirs(destination, exist_ok=True)
        sb = sciencebasepy.SbSession()
        sb.download_file(sb_url, prms_predictions+'.zip', destination)
    with zipfile.ZipFile(os.path.join(destination,prms_predictions+'.zip'), 'r') as zip_ref:
        zip_ref.extractall(destination)
        


def fill_discharge_prms(site_num, fill_segment_crosswalk, destination, prms_predictions):
    '''reads in prms predictions of discharge and fills nan gaps for trenton and schuylkill nwis time series
    :fill_segments: [dict] dictionary of sites to fill data for; each site in the dictionary contains 'nwis_site' (nwis site number) and 'seg_id_nat' (segment id closest to discharge location)
    :destination: [str] directory where prms data file is located
    :prms_predictions: [str] name of prms file'''
    
    sn_temp_data = pd.read_csv(os.path.join(destination,prms_predictions+'.csv'), parse_dates = True, index_col = 'date')
    
    sn_temp_site = sn_temp_data[sn_temp_data['seg_id_nat'] == fill_segment_crosswalk[site_num]]
    
    nwis_data_path = os.path.join('02_munge', 'out', 'D', 'usgs_nwis_{}.csv'.format(site_num))
    nwis_site_data = pd.read_csv(nwis_data_path, parse_dates = True, index_col = 'datetime')
    
    #fill the gaps make sure to from cubic meters to cfs
    nwis_site_data['filled'] = nwis_site_data['discharge'].fillna(sn_temp_site['seg_outflow']*35.315)
    
    #drop the old discharge and rename filled as discharge
    nwis_site_data = nwis_site_data.drop(columns = 'discharge')
    nwis_site_data.rename(columns = {'filled':'discharge'}, inplace=True)
    
    #write to dataframe
    nwis_site_data.to_csv(nwis_data_path)


def fill_single_site_data(site_num):
    # site data comes in from snakemake as a set, get single value from set
    if type(site_num)==set:
        site_num = list(site_num)[0]
    
    sb_url = config['sb_url']
    prms_predictions = config['prms_predictions']
    destination = config['destination']
    fill_segment_crosswalk = config['fill_segments']
    
    # if we don't already have sntemp data, download it
    if f'{prms_predictions}.csv' not in os.listdir(destination):
        download_unzip_sb(sb_url, prms_predictions, destination)

    fill_discharge_prms(site_num, fill_segment_crosswalk, destination, prms_predictions)

def fill_all_sites_data():
    with open("01_fetch/wildcards_fetch_config.yaml", 'r') as stream:
        site_ids = yaml.safe_load(stream)['fetch_usgs_nwis.py']['sites']
    for site_num in site_ids:
        fill_single_site_data(site_num)

if __name__ == '__main__':
    fill_all_sites_data()
