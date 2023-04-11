import os
import numpy as np
import pandas as pd
import boto3
import yaml
from scipy import signal
import sys
sys.path.insert(0, os.path.join('01_fetch', 'src'))
import utils

# import config
with open("02_munge/params_config_munge_noaa_nos.yaml", 'r') as stream:
    config = yaml.safe_load(stream)

# check where to read data inputs from
read_location = config['read_location']

# set up write location data outputs
write_location = config['write_location']
s3_client = utils.prep_write_location(write_location, config['aws_profile'])
s3_bucket = config['s3_bucket']

def get_datafile_list(station_id, read_location, s3_client=None, s3_bucket=None):
    raw_datafiles = {}
    if read_location=='S3':
        raw_datafiles = [obj['Key'] for obj in s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=f'01_fetch/out/noaa_nos_{station_id}')['Contents']]
    elif read_location=='local':
        prefix = os.path.join('01_fetch', 'out')
        file_prefix=f'noaa_nos_{station_id}'
        raw_datafiles = [os.path.join(prefix, f) for f in os.listdir(prefix) if f.startswith(file_prefix)]
    return raw_datafiles

def read_data(raw_datafile):
    if read_location == 'local':
        print(f'reading data from local: {raw_datafile}')
        # read in raw data as pandas df
        df = pd.read_csv(raw_datafile)
    elif read_location == 'S3':
        print(f'reading data from s3: {raw_datafile}')
        obj = s3_client.get_object(Bucket=s3_bucket, Key=raw_datafile)
        # read in raw data as pandas df
        df = pd.read_csv(obj.get("Body"))
    return df

def fill_gaps(x):
    '''fills any data gaps in the middle of the input series
    using linear interpolation; returns gap-filled time series
    '''
    #find nans
    bd = np.isnan(x)

    #early exit if there are no nans  
    if not bd.any():
        return x

    #find nonnans index numbers
    gd = np.flatnonzero(~bd)

    #ignore leading and trailing nans
    bd[:gd.min()]=False
    bd[(gd.max()+1):]=False

    #interpolate nans
    x[bd] = np.interp(np.flatnonzero(bd),gd,x[gd])
    return x


def butterworth_filter(df, butterworth_filter_params):
    #parameter for butterworth filter
    # filter order
    order_butter = butterworth_filter_params['order_butter']
    # cutoff frequency
    fc= butterworth_filter_params['fc']
    # sample interval
    fs = butterworth_filter_params['fs']
    # filter accepts 1d array
    prod = butterworth_filter_params['product']

    # get only product of interest 
    x = df[prod]

    # apply butterworth filter
    b,a = signal.butter(order_butter, fc, 'low', fs=fs, output='ba')
    x_signal = signal.filtfilt(b,a,x[x.notnull()])
    df.loc[x.notnull(), prod+'_filtered'] = x_signal
    return df

def process_data_to_csv(site, site_raw_datafiles, qa_to_drop, flags_to_drop_by_var, agg_level, prop_obs_required, butterworth_filter_params):
    '''
    process raw data text files into clean csvs, including:
        dropping unwanted flags
        converting datetime column to datetime format
        converting all data columns to numeric type
        removing metadata columns so that only datetime and data columns remain 
    '''
    print(f'processing and saving locally')
    combined_df = pd.DataFrame(columns=['datetime'])
    for raw_datafile in site_raw_datafiles:
        # read in data file (unless it is empty)
        try:
            df = read_data(raw_datafile)
        except pd.errors.EmptyDataError:
            continue
    
        # if there is a SD column, drop it
        if 's' in df.columns:
            df = df.drop(['s'], axis=1)

        # if there is a QA column, drop any QA/QC flags that we don't want to include
        if 'q' in df.columns:
            df['v'] = np.where(df['q'].isin(qa_to_drop), np.nan, df['v'])
            # drop QA/QC flag column
            df = df.drop(['q'], axis=1)

        # get the name of the variable we are processing to inform next processing steps
        var_name = os.path.splitext(os.path.basename(raw_datafile))[0].replace(f'noaa_nos_{site}_', '')

        # drop any flagged we don't want to include
        if 'f' in df.columns:
            flags_to_drop = flags_to_drop_by_var[var_name]
            for i in range(len(flags_to_drop)):
                drop_flag = flags_to_drop[i]
                if drop_flag:
                    df['v'] = np.where(df['f'].str.split(',', expand=True)[i]=='1', np.nan, df['v'])
            # drop flag column
            df = df.drop(['f'], axis=1)

        # replace value column 'v' with variable name
        df = df.rename(columns={'v': var_name, 't': 'datetime'})

        # convert datetime column to datetime type
        df['datetime'] = df['datetime'].astype('datetime64')

        # add to combined df
        combined_df = combined_df.merge(df, on='datetime', how='outer')

    # sort by datetime
    combined_df.sort_values('datetime', inplace=True)

    # make all other columns numeric
    cols = combined_df.columns.drop('datetime')
    combined_df[cols] = combined_df[cols].apply(pd.to_numeric, errors='coerce')

    # aggregate data to specified timestep
    combined_df = utils.process_to_timestep(combined_df, cols, agg_level, prop_obs_required)

    # drop any columns with no data
    combined_df.dropna(axis=1, how='all', inplace=True)

    # fill inner data gaps on water level using linear interpolation
    # so we can apply the butterworth filter
    combined_df['water_level'] = fill_gaps(combined_df['water_level'])
    
    # apply butterworth filter
    butterworth_df = butterworth_filter(combined_df, butterworth_filter_params)

    # save pre-processed data
    dir = os.path.join('.', '02_munge',  'out', agg_level)
    if not os.path.exists(dir): os.mkdir(dir)
    data_outfile_csv = os.path.join(dir, f'noaa_nos_{site}.csv')
    butterworth_df.to_csv(data_outfile_csv, index=True)
    
    if write_location == 'S3':
        print('uploading to s3')
        s3_client.upload_file(data_outfile_csv, s3_bucket, local_to_s3_pathname(data_outfile_csv))
        
    return butterworth_df

def extract_daily_tidal_data(hourly_tidal_data, site):
    #calling it datetime to match other data sources, it is just the date
    hourly_tidal_data['datetime'] = hourly_tidal_data.index.date
    hourly_tidal_data['obs_pred'] = hourly_tidal_data['water_level'] - hourly_tidal_data['predictions']
    d = {
    'wl_range':hourly_tidal_data.groupby(hourly_tidal_data.datetime)['water_level'].max() -  hourly_tidal_data.groupby(hourly_tidal_data.datetime)['water_level'].min(),
    'wl_max' : hourly_tidal_data.groupby(hourly_tidal_data.datetime)['water_level'].max(),
    'wl_obs_pred':hourly_tidal_data.groupby(hourly_tidal_data.datetime)['obs_pred'].sum(),
    'wl_filtered':hourly_tidal_data.groupby(hourly_tidal_data.datetime)['water_level_filtered'].mean(),
    'air_pressure': hourly_tidal_data.groupby(hourly_tidal_data.datetime)['air_pressure'].mean()
    #'air_temperature': hourly_tidal_data.groupby(hourly_tidal_data.datetime)['air_temperature'].mean()
    }

    daily_df = pd.DataFrame(data = d, index = d['wl_range'].index)
    
    if 'conductivity' in hourly_tidal_data.columns:
        d['conductivity'] = hourly_tidal_data.groupby(hourly_tidal_data.datetime)['conductivity'].mean()
    
    
    # save pre-processed data
    print("processed site "+ site +" to daily time step")
    dir = os.path.join('.', '02_munge',  'out', 'daily_summaries')
    if not os.path.exists(dir): os.mkdir(dir)
    data_outfile_csv = os.path.join(dir, f'noaa_nos_{site}.csv')
    daily_df.to_csv(data_outfile_csv, index=True)
    
    if write_location == 'S3':
        print('uploading to s3')
        s3_client.upload_file(data_outfile_csv, s3_bucket, local_to_s3_pathname(data_outfile_csv))
        
    return daily_df


def munge_single_site_data(site_num):
    # site data comes in from snakemake as a set, get single value from set
    if type(site_num)==set:
        site_num = list(site_num)[0] 
    # determine which data flags we want to drop
    flags_to_drop_by_var = config['flags_to_drop']
    # determine which QA/QC levels we want to drop
    qa_to_drop = config['qa_to_drop']
    # number of measurements required to consider average valid
    prop_obs_required = config['prop_obs_required']
    # timestep to aggregate to
    agg_level = config['agg_level']

    # butterworth filter parameters
    butterworth_filter_params = config['butterworth_filter_params']
    # process raw data files into csv
    site_raw_datafiles = get_datafile_list(site_num, read_location=read_location)
    if bool(site_raw_datafiles):
        df = process_data_to_csv(site_num, site_raw_datafiles, qa_to_drop, flags_to_drop_by_var, agg_level, prop_obs_required, butterworth_filter_params)
        daily_df = extract_daily_tidal_data(df, site_num)
    else:
        print('no data in 01_fetch/out/ for site '+ site_num)

def munge_all_sites_data():
    with open("01_fetch/wildcards_fetch_config.yaml", 'r') as stream:
        site_ids = yaml.safe_load(stream)['fetch_noaa_nos.py']['sites']
    for site_num in site_ids:
        munge_single_site_data(site_num)

if __name__ == '__main__':
    munge_all_sites_data()
