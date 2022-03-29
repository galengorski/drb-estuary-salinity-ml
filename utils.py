import sys
import boto3
import subprocess
import pandas as pd
import os

def prep_write_location(write_location, aws_profile):
    if write_location=='S3':
        cont = input("You are about to write to S3, and you may overwrite existing data. Are you sure you want to do this? (yes, no)")
        if cont=="no":
            sys.exit("Aborting data fetch.")
    # after you have configured saml2aws, you can log in and create a new 
    # token by executing the following command:
    # keep all of the s3 arguments hard coded as I don't see us changing them much
    subprocess.run(["saml2aws", "login", "--skip-prompt", "--role", "arn:aws:iam::807615458658:role/adfs-wma-developer"])
    
    # start S3 session so that we can upload data
    session = boto3.Session(profile_name=aws_profile)
    s3_client = session.client('s3')
    return s3_client

def usgs_nwis_param_code_to_name(code):
    '''process usgs nwis parameter code to machine+human-readable name string'''
    # read in parameter metadata df
    params_df = pd.read_csv(os.path.join('.', '01_fetch', 'out', 'metadata', 'usgs_nwis_params.csv'), dtype={"parm_cd":"string"})
    # find the corresponding parameter name
    full_name = params_df[params_df['parm_cd']==code]['parm_nm'].iloc[0]
    # give it a shorter machine-readable name
    name = full_name.split(',')[0].replace(' ', '_').lower()
    return name

def process_to_timestep(df, cols, agg_level, prop_obs_required):
    '''
    aggregate df to specified timestep
    must have datetimes in a column named 'datetime'
    '''
    # get proportion of measurements available for timestep
    expected_measurements = df.resample(agg_level, on='datetime').count().mode()[cols].loc[0]
    observed_measurements = df.resample(agg_level, on='datetime').count()[cols].loc[:]
    prop_df = observed_measurements / expected_measurements
    # calculate averages for timestep
    df = df.resample(agg_level, on='datetime').mean()
    # only keep averages where we have enough measurements
    df.where(prop_df.gt(prop_obs_required), inplace=True)
    return df

def local_to_s3_pathname(local_pathname):
    '''
    takes a local file path name and converts it to the
    properly formatted s3 file path name
    '''
    return local_pathname.replace('.\\','').replace('\\', '/')