import os
import numpy as np
import pandas as pd
import re
import boto3
import sys

def prep_write_location(write_location):
    if write_location=='S3':
        cont = input("You are about to write to S3, and you may overwrite existing data. Are you sure you want to do this? (yes, no)")
        if cont=="no":
            sys.exit("Aborting data fetch.")
    # start S3 session so that we can upload data
    session = boto3.Session(profile_name='dev')
    s3_client = session.client('s3')
    return s3_client

def process_params_to_csv(raw_params_txt, params_outfile_csv, write_location, s3_client):
    '''process raw parameter text file into a csv file'''
    print('reading raw parameter data from s3')
    obj = s3_client.get_object(Bucket='drb-estuary-salinity', Key=raw_params_txt)
    params_df = pd.read_csv(obj.get("Body"), comment='#', sep='\t', lineterminator='\n')
    print('processing parameter file and saving locally')
    params_df.drop(index=0, inplace=True)
    params_df.to_csv(params_outfile_csv)
    if write_location == 'S3':
        print('uploading to s3')
        s3_client.upload_file(params_outfile_csv, 'drb-estuary-salinity', '02_munge/out/'+os.path.basename(params_outfile_csv))

def process_data_to_csv(raw_datafile, flags_to_drop, agg_level, prop_obs_required, write_location, s3_client):
    '''
    process raw data text files into clean csvs, including:
        dropping unwanted flags
        converting datetime column to datetime format
        converting all data columns to numeric type
        removing metadata columns so that only datetime and data columns remain 
    '''
    print(f'reading data from s3: {raw_datafile}')
    obj = s3_client.get_object(Bucket='drb-estuary-salinity', Key=raw_datafile)
    # read in raw data as pandas df
    df = pd.read_csv(obj.get("Body"), comment='#', sep='\t', lineterminator='\n', low_memory=False)
    
    print(f'processing and saving locally')
    # drop first row which does not contain useful headers or data
    df.drop(index=0, inplace=True)

    # replace all flagged data we want to remove with NaN
    flag_cols = [col for col in df.columns if re.match("[0-9]+_[0-9]+_[a-z]+$", col)]
    for flag_col in flag_cols:
        flag_data_col = flag_col[:-3]
        df[flag_data_col] = np.where(df[flag_col].isin(flags_to_drop), np.nan, df[flag_data_col])

    # drop site info and flag columns
    df = df.drop(['agency_cd', 'site_no', 'tz_cd']+flag_cols, axis=1)

    # convert datetime column to datetime type
    df['datetime'] = df['datetime'].astype('datetime64')

    # make all other columns numeric
    cols = df.columns.drop('datetime')
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')

    # aggregate data to specified timestep
    if agg_level == 'daily':
        # get proportion of daily measurements available
        prop_df = df.groupby([df['datetime'].dt.date]).count()[cols].div(df.groupby([df['datetime'].dt.date]).count()['datetime'], axis=0)
        # calculate daily averages
        df = df.groupby([df['datetime'].dt.date]).mean()
    # only keep averages where we have enough measurements
    df.where(prop_df.gt(prop_obs_required))

    # save pre-processed data
    data_outfile_csv = os.path.join('.', '02_munge', 'out', os.path.splitext(os.path.basename(raw_datafile))[0]+'.csv')
    df.to_csv(data_outfile_csv, index=True)
    
    if write_location == 'S3':
        print('uploading to s3')
        s3_client.upload_file(data_outfile_csv, 'drb-estuary-salinity', '02_munge/out/'+os.path.basename(data_outfile_csv))

def main():
    # choose where you want to write your data outputs: local or S3
    write_location = 'local'
    s3_client = prep_write_location(write_location)

    # process raw parameter data into csv
    raw_params_txt = '01_fetch/out/usgs_nwis_params.txt'
    params_outfile_csv = os.path.join('.', '02_munge', 'out', 'usgs_nwis_params.csv')
    process_params_to_csv(raw_params_txt, params_outfile_csv, write_location, s3_client)

    # get list of raw data files to process
    raw_datafiles = [obj['Key'] for obj in s3_client.list_objects_v2(Bucket='drb-estuary-salinity', Prefix='01_fetch/out/usgs_nwis_0')['Contents']]

    # determine which data flags we want to drop
    # e     Value has been edited or estimated by USGS personnel and is write protected
    # &     Value was computed from affected unit values
    # E     Value was computed from estimated unit values.
    # A     Approved for publication -- Processing and review completed.
    # P     Provisional data subject to revision.
    # <     The value is known to be less than reported value and is write protected.
    # >     The value is known to be greater than reported value and is write protected.
    # 1     Value is write protected without any remark code to be printed
    # 2     Remark is write protected without any remark code to be printed
    flags_to_drop = ['e', '&', 'E', 'P', '<', '>', '1', '2']

    # number of measurements required to consider average valid
    # we will assume that we need half of the timestep measurements
    prop_obs_required = 0.5

    # timestep to aggregate to
    # options: daily
    agg_level = 'daily'

    # process raw data files into csv
    for raw_datafile in raw_datafiles:
        process_data_to_csv(raw_datafile, flags_to_drop, agg_level, prop_obs_required, write_location, s3_client)

if __name__ == '__main__':
    main()