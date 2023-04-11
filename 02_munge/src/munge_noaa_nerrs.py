import os
import pandas as pd
import numpy as np
import yaml
import sys
sys.path.insert(0, os.path.join('01_fetch', 'src'))
import utils

# import config
with open("02_munge/params_config_munge_noaa_nerrs.yaml", 'r') as stream:
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
        file_prefix=f'noaa_nerrs_{station_id}'
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

def process_data_to_csv(site, site_raw_datafiles, column_mapping, flags_to_drop, agg_level, prop_obs_required, fill_anom_w_nos):
    '''
    process raw data text files into clean csvs, including:
        dropping unwanted flags
        converting datetime column to datetime format
        converting all data columns to numeric type
        removing metadata columns so that only datetime and data columns remain 
    '''
    for raw_datafile in site_raw_datafiles:
        year_df = read_data(raw_datafile)
        if raw_datafile == site_raw_datafiles[0]:
            combined_df = year_df.copy()
        else:
            combined_df = pd.concat([combined_df, year_df], ignore_index=True)

    # filter out data that we don't want
    col_values_accepted = config['col_values_accepted']
    for col in col_values_accepted.keys():
        combined_df = combined_df[combined_df[col].isin(col_values_accepted[col])]
        combined_df.drop(col, axis=1, inplace=True)

    # replace all flagged data we want to remove with NaN
    vars_to_keep = list(column_mapping.keys())
    vars_to_keep.remove('DatetimeStamp')
    for var in vars_to_keep:
        flag_col = f'F_{var}'
        combined_df[var] = np.where(combined_df[flag_col].isin(flags_to_drop), np.nan, combined_df[var])

    # drop any columns we don't want
    combined_df = combined_df[column_mapping.keys()]

    # map column names
    combined_df.rename(columns=column_mapping, inplace=True)

    # convert datetime column to datetime type
    combined_df['datetime'] = combined_df['datetime'].astype('datetime64')

    # make all other columns numeric
    cols = combined_df.columns.drop('datetime')
    combined_df[cols] = combined_df[cols].apply(pd.to_numeric, errors='coerce')

    # aggregate data to specified timestep
    combined_df = utils.process_to_timestep(combined_df, column_mapping.values(), agg_level, prop_obs_required)

    # drop any columns with no data
    combined_df.dropna(axis=1, how='all', inplace=True)
    
    # there are some anomalous values below 980
    combined_df.loc[combined_df['air_pressure'] < 980,'air_pressure'] = np.nan
    
    if fill_anom_w_nos:
        try:
            noaa_nos_file = os.listdir('02_munge/out/daily_summaries')
            noaa_nos = pd.read_csv(os.path.join('02_munge','out','daily_summaries',noaa_nos_file[0]), index_col = 'datetime')
            combined_df['air_pressure'] = combined_df['air_pressure'].fillna(noaa_nos['air_pressure'])
        except:
            print('No NOAA NOS data to fill NERR air pressure record')
    
    #fill gaps in temperature, air pressure, wind speed and wind direction
    # all < 0.5 %, precipitation has more like 10% missing so we won't fill that
    combined_df['temperature'] = fill_gaps(combined_df['temperature'])
    combined_df['air_pressure'] = fill_gaps(combined_df['air_pressure'])
    combined_df['wind_speed'] = fill_gaps(combined_df['wind_speed'])
    combined_df['wind_direction'] = fill_gaps(combined_df['wind_direction'])
    combined_df['wind_speed_direction'] = -1*combined_df['wind_speed']*np.cos(combined_df['wind_direction']*(np.pi/180))

    # save pre-processed data
    data_outfile_csv = os.path.join('.', '02_munge', 'out', agg_level, 'noaa_nerrs_delsjmet.csv')
    combined_df.to_csv(data_outfile_csv, index=True)

    if write_location == 'S3':
        print('uploading to s3')
        s3_client.upload_file(data_outfile_csv, s3_bucket, utils.local_to_s3_pathname(data_outfile_csv))

    return combined_df

def munge_single_site_data(site_num):
    # site data comes in from snakemake as a set, get single value from set
    if type(site_num)==set:
        site_num = list(site_num)[0]

    # get variables we want to process
    column_mapping = config['vars']
    # determine which data flags we want to drop
    flags_to_drop = config['flags_to_drop']
    # timestep to aggregate to
    agg_level = config['agg_level']
    # number of measurements required to consider average valid
    prop_obs_required = config['prop_obs_required']
    #should air pressure anomalies in NERR be filled with NOS data
    fill_anom_w_nos = config['fill_anom_w_nos']

    # make output directories if they don't exist
    os.makedirs('02_munge/out', exist_ok=True)
    os.makedirs(f'02_munge/out/{agg_level}', exist_ok=True)

    # process raw data files into csv
    site_raw_datafiles = get_datafile_list(site_num, read_location=read_location)
    
    process_data_to_csv(site_num, site_raw_datafiles, column_mapping, flags_to_drop, agg_level, prop_obs_required, fill_anom_w_nos)

def munge_all_sites_data():
    with open("01_fetch/wildcards_fetch_config.yaml", 'r') as stream:
        site_ids = yaml.safe_load(stream)['fetch_noaa_nerrs.py']['sites']
    for site_num in site_ids:
        munge_single_site_data(site_num)

if __name__ == '__main__':
    munge_all_sites_data()