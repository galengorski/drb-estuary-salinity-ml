import os
import numpy as np
import pandas as pd
import re
import boto3

# get dictionary of all possible USGS site parameters
def process_params_to_csv(raw_params_txt, params_outfile_csv, s3_client):
    '''process raw parameter text file into a csv file'''
    print('processing parameter file and saving locally')
    params_df = pd.read_csv(raw_params_txt, comment='#', sep='\t', lineterminator='\n')
    params_df.drop(index=0, inplace=True)
    params_df.to_csv(params_outfile_csv)
    print('uploading to s3')
    s3_client.upload_file(params_outfile_csv, 'drb-estuary-salinity', '01_munge/out/'+os.path.basename(params_outfile_csv))

def process_data_to_csv(raw_datafile, flags_to_drop, s3_client):
    '''
    process raw data text files into clean csvs, including:
        dropping unwanted flags
        converting datetime column to datetime format
        converting all data columns to numeric type
        removing metadata columns so that only datetime and data columns remain 
    '''
    print(f'processing {raw_datafile} and saving locally')
    # read in raw data as pandas df
    df = pd.read_csv(raw_datafile, comment='#', sep='\t', lineterminator='\n', low_memory=False)
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

    # aggregate data to daily timestep
    # get proportion of daily measurements available
    prop_df = df.groupby([df['datetime'].dt.date]).count()[cols].div(df.groupby([df['datetime'].dt.date]).count()['datetime'], axis=0)
    # calculate daily averages
    df = df.groupby([df['datetime'].dt.date]).mean()
    # only keep daily averages where we have enough measurements
    df.where(prop_df.gt(prop_obs_required))

    # save pre-processed data
    data_outfile_csv = os.path.join('.', '02_munge', 'out', os.path.splitext(os.path.basename(raw_datafile))[0]+'.csv')
    df.to_csv(data_outfile_csv, index=True)

    print('uploading to s3')
    s3_client.upload_file(data_outfile_csv, 'drb-estuary-salinity', '02_munge/out/'+os.path.basename(data_outfile_csv))

def main():
    # start S3 session so that we can upload data
    session = boto3.Session(profile_name='dev')
    s3_client = session.client('s3')

    # process raw parameter data into csv
    raw_params_txt = os.path.join('.', '01_fetch', 'out', 'params.txt')
    params_outfile_csv = os.path.join('.', '02_munge', 'out', 'params.csv')
    process_params_to_csv(raw_params_txt, params_outfile_csv, s3_client)

    # get list of raw data files to process
    raw_datafiles = [os.path.join('.', '01_fetch', 'out', file) for file in os.listdir(os.path.join('.', '01_fetch', 'out'))]
    raw_datafiles.remove(os.path.join('.', '01_fetch', 'out', 'params.txt'))

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

    # number of measurements required to consider daily average valid
    # we will assume that we need half of the daily measurements
    prop_obs_required = 0.5

    # process raw data files into csv
    for raw_datafile in raw_datafiles:
        process_data_to_csv(raw_datafile, flags_to_drop, s3_client)

if __name__ == '__main__':
    main()