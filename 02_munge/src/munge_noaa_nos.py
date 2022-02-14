import os
import numpy as np
import pandas as pd
import re
import boto3

def combine_to_one_csv(station_id, product):
    #define key to point to datasets for specified station
    key = f'01_fetch/out/noaa_nos_{station_id}_{product}'
    #create a list of csv files in s3 bucket with observational data specific to station
    prefix_df = []
    prefix_obs = [obj['Key'] for obj in s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=f'01_fetch/out/noaa_nos_{station_id}_{product}')['Contents']]
    #grab each dataset from bucket and combine to a single csv file
        for obs in prefix_obs:
            obj = s3_client.get_object(Bucket=s3_bucket, Key=obs)
            temp = pd.read_csv(obj.get('Body'), index_col=None, header=0, encoding='utf8')
            prefix_df.append(temp)
    #sort columns to timestep
    df = pd.concat(prefix_df, ignore_index=True, sort = True)
    data_outfile_csv = f'{station_id}_{product}'
    #save to csv file
    if write_location == 'S3':
           print('uploading to s3')
           s3_client.upload_file(data_outfile_csv, s3_bucket, '02_munge/out/'+os.path.basename(data_outfile_csv))