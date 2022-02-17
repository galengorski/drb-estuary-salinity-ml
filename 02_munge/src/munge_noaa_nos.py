import os
import numpy as np
import pandas as pd
import boto3
import yaml
import utils

def read_data(raw_datafile, read_location, s3_bucket):
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

def process_data_to_csv(site, site_raw_datafiles, qa_to_drop, flags_to_drop_by_var, agg_level, prop_obs_required, read_location, write_location, s3_bucket, s3_client):
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
        # read in data file
        df = read_data(raw_datafile, read_location, s3_bucket)
    
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

    # save pre-processed data
    data_outfile_csv = os.path.join('.', '02_munge', 'out', f'noaa_nos_{site}.csv')
    combined_df.to_csv(data_outfile_csv, index=True)
    
    if write_location == 'S3':
        print('uploading to s3')
        s3_client.upload_file(data_outfile_csv, s3_bucket, '02_munge/out/'+os.path.basename(data_outfile_csv))

def main():
    # import config
    with open("02_munge/munge_config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)['munge_noaa_nos.py']

    # check where to read data inputs from
    read_location = config['read_location']

    # set up write location data outputs
    write_location = config['write_location']
    s3_client = utils.prep_write_location(write_location, config['aws_profile'])
    s3_bucket = config['s3_bucket']

    # get list of raw data files to process, by site
    with open("01_fetch/fetch_config.yaml", 'r') as stream:
        station_ids = yaml.safe_load(stream)['fetch_noaa_nos.py']['station_ids']
    raw_datafiles = {}
    for station_id in station_ids:
        raw_datafiles[station_id] = [obj['Key'] for obj in s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=f'01_fetch/out/noaa_nos_{station_id}')['Contents']]
    
    # determine which data flags we want to drop
    flags_to_drop_by_var = config['flags_to_drop']
    # determine which QA/QC levels we want to drop
    qa_to_drop = config['qa_to_drop']
    # number of measurements required to consider average valid
    prop_obs_required = config['prop_obs_required']
    # timestep to aggregate to
    agg_level = config['agg_level']

    # process raw data files into csv
    for site, site_raw_datafiles in raw_datafiles.items():
        df = process_data_to_csv(site, site_raw_datafiles, qa_to_drop, flags_to_drop_by_var, agg_level, prop_obs_required, read_location, write_location, s3_bucket, s3_client)
        # apply butterworth filter

if __name__ == '__main__':
    main()