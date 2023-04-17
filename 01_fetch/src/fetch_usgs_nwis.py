import os
import urllib
import yaml
import datetime
import pandas as pd
import sys
sys.path.insert(0, os.path.join('01_fetch', 'src'))
import utils

# import config
with open("01_fetch/params_config_fetch_usgs_nwis.yaml", 'r') as stream:
    config = yaml.safe_load(stream)

# set up write location data outputs
write_location = config['write_location']
s3_client = utils.prep_write_location(write_location, config['aws_profile'])
s3_bucket = config['s3_bucket']

def fetch_site_info(site_num, outfile):
    '''fetch USGS NWIS data for locations in site_list (gets all parameters available)'''
    site_url = f'https://waterservices.usgs.gov/nwis/site/?format=rdb&sites={site_num}&seriesCatalogOutput=true'
    print(f'fetching site info for site {site_num} and saving locally')
    urllib.request.urlretrieve(site_url, outfile)

def process_site_info_to_csv(raw_site_info_txt, site_info_outfile_csv):
    '''
    process raw site info text file into a csv file,
    return minimum date of measured data (for any parameter) as start date for site
    '''
    print(f'processing raw site info to get start date for site')
    site_info_df = pd.read_csv(raw_site_info_txt, comment='#', sep='\t', lineterminator='\n')
    site_info_df.drop(index=0, inplace=True)

    # get subset of unit values, other codes listed below:
    # “dv” (daily values)
    # “uv” or “iv” (unit values)
    # “qw” (water-quality)
    # “sv” (sites visits)
    # “pk” (peak measurements)
    # “gw” (groundwater levels)
    # “ad” (sites included in USGS Annual Water Data Reports External Link)
    # “aw” (sites monitored by the USGS Active Groundwater Level Network External Link)
    # “id” (historical instantaneous values)
    site_info_df_subset = site_info_df.loc[(site_info_df['data_type_cd']=='uv') | (site_info_df['data_type_cd']=='iv'), :]
    # subset to only columns of interest
    site_info_df_subset = site_info_df_subset[['parm_cd', 'begin_date', 'end_date', 'count_nu']]
    # save this table of info
    site_info_df_subset.to_csv(site_info_outfile_csv, index=False)
    if write_location == 'S3':
        print('uploading to s3')
        s3_client.upload_file(site_info_outfile_csv, s3_bucket, local_to_s3_pathname(site_info_outfile_csv))

def fetch_params(outfile):
    '''get table of all possible USGS site parameters'''
    params_url = 'https://help.waterdata.usgs.gov/code/parameter_cd_query?fmt=rdb&group_cd=%'
    print('fetching parameter file and saving locally')
    urllib.request.urlretrieve(params_url, outfile)

def process_params_to_csv(raw_params_txt, params_outfile_csv):
    '''process raw parameter text file into a csv file'''
    print('reading raw parameter data from local')
    params_df = pd.read_csv(raw_params_txt, comment='#', sep='\t', lineterminator='\n')
    print('processing parameter file and saving locally')
    params_df.drop(index=0, inplace=True)
    params_df.to_csv(params_outfile_csv, index=False)
    if write_location == 'S3':
        print('uploading to s3')
        s3_client.upload_file(params_outfile_csv, bucket, local_to_s3_pathname(params_outfile_csv))
    return params_df

def fetch_data(site_num, start_dt, end_dt, outfile):
    '''fetch USGS NWIS data for locations in site_list (gets all parameters available)'''
    data_url = f'https://waterservices.usgs.gov/nwis/iv?format=rdb&sites={site_num}&startDT={start_dt}&endDT={end_dt}'
    print(f'fetching data for site {site_num} and saving locally')
    urllib.request.urlretrieve(data_url, outfile)
    if write_location == 'S3':
        print('uploading to s3')
        s3_client.upload_file(outfile, s3_bucket, local_to_s3_pathname(outfile))

def fetch_param_file():
    # fetch parameter file
    params_outfile_txt = os.path.join('.', '01_fetch', 'out', 'metadata', 'usgs_nwis_params.txt')
    fetch_params(params_outfile_txt)
    # process raw parameter data into csv
    params_outfile_csv = os.path.join('.', '01_fetch', 'out', 'metadata', 'usgs_nwis_params.csv')
    params_df = process_params_to_csv(params_outfile_txt, params_outfile_csv)

def fetch_single_site_data(site_num):
    # make output directories if they don't exist
    os.makedirs('01_fetch/out', exist_ok=True)
    os.makedirs('01_fetch/out/metadata', exist_ok=True)

    # site data comes in from snakemake as a set, get single value from set
    if type(site_num)==set:
        site_num = list(site_num)[0]

    # fetch raw data files
    site_info_outfile_txt = os.path.join('.', '01_fetch', 'out', 'metadata', f'usgs_nwis_site_info_{site_num}.txt')
    fetch_site_info(site_num, site_info_outfile_txt)
    site_info_outfile_csv = os.path.join('.', '01_fetch', 'out', 'metadata', f'usgs_nwis_site_info_{site_num}.csv')
    process_site_info_to_csv(site_info_outfile_txt, site_info_outfile_csv)
    start_dt = config['start_dt']
    end_dt = config['end_dt']

    # start and end dates for data fetch
    data_outfile_txt = os.path.join('.', '01_fetch', 'out', f'usgs_nwis_{site_num}.txt')
    fetch_data(site_num, start_dt, end_dt, data_outfile_txt)

def fetch_all_sites_data():
    with open("01_fetch/wildcards_fetch_config.yaml", 'r') as stream:
        site_ids = yaml.safe_load(stream)['fetch_usgs_nwis.py']['sites']
    for site_num in site_ids:
        fetch_single_site_data(site_num)
    # fetch parameter file
    fetch_param_file()

if __name__ == '__main__':
    fetch_all_sites_data()