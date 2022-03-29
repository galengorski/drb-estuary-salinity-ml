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
        


def fill_discharge_prms(fill_segments, destination, prms_predictions):
    '''reads in prms predictions of discharge and fills nan gaps for trenton and schuylkill nwis time series
    :fill_segments: [dict] dictionary of sites to fill data for; each site in the dictionary contains 'nwis_site' (nwis site number) and 'seg_id_nat' (segment id closest to discharge location)
    :destination: [str] directory where prms data file is located
    :prms_predictions: [str] name of prms file'''
    
    sn_temp_data = pd.read_csv(os.path.join(destination,prms_predictions+'.csv'), parse_dates = True, index_col = 'date')
    
for segment_details in fill_segments.values():
    sn_temp_site = sn_temp_data[sn_temp_data['seg_id_nat'] == segment_details['seg_id_nat']]

    nwis_data_path = os.path.join('02_munge', 'out', 'D', 'usgs_nwis_{}.csv'.format(segment_details['nwis_site']))
    nwis_site_data = pd.read_csv(nwis_data_path, parse_dates = True, index_col = 'datetime')

    #fill the gaps make sure to from cubic meters to cfs
    nwis_site_data['filled'] = nwis_site_data['discharge'].fillna(sn_temp_site['seg_outflow']*35.315)

    #drop the old discharge and rename filled as discharge
    nwis_site_data = nwis_site_data.drop(columns = 'discharge')
    nwis_site_data.rename(columns = {'filled':'discharge'}, inplace=True)

    #write to dataframe
    nwis_site_data.to_csv(nwis_data_path)


def main():
    # import config
    with open("02_munge/munge_config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)['fill_discharge_prms.py']
    
    sb_url = config['sb_url']
    prms_predictions = config['prms_predictions']
    destination = config['destination']
    fill_segments = config['fill_segments']
    
    download_unzip_sb(sb_url, prms_predictions, destination)
    fill_discharge_prms(fill_segments, destination, prms_predictions)

if __name__ == '__main__':
    main()
