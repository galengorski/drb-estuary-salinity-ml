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
    expected_measurements = df.resample(agg_level, on='datetime').count().mode().loc[0]
    observed_measurements = df.resample(agg_level, on='datetime').count().loc[:]
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

def download_s3_to_local(s3_dir_prefix, local_outdir, file_id):
    '''download data files from s3 bucket to local machine for development
    file_id - a file identifier substring that is contained within all 
    the file names you want to download. For example 'usgs_nwis' will 
    download all files with 'usgs_nwis' in the file name'''
    
    # assumes we are using a credential profile names 'dev'
    write_location = 'local'
    aws_profile = 'dev'
    s3_client = prep_write_location(write_location, aws_profile)
    # end the name of the bucket you want to read/write to:
    s3_bucket = 'drb-estuary-salinity'
    
    # create the output file directory on your local
    os.makedirs(local_outdir, exist_ok=True)

    # loop through all objects with this prefix that contain .csv and file_id and download
    for obj in s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=s3_dir_prefix)['Contents']:
        s3_fpath = obj['Key']
        if ".csv" and file_id not in s3_fpath:
            continue
        local_fpath = os.path.join(local_outdir,obj['Key'].split('/')[2])
        s3_client.download_file(s3_bucket, s3_fpath, local_fpath)
        print(s3_fpath+' Downloaded to local')

def get_datafile_list(read_location, s3_client=None, s3_bucket=None):
    raw_datafiles = {}
    if read_location=='S3':
        raw_datafiles = [obj['Key'] for obj in s3_client.list_objects_v2(Bucket=s3_bucket, Prefix='01_fetch/out/usgs_nwis_0')['Contents']]
    elif read_location=='local':
        prefix = os.path.join('01_fetch', 'out')
        file_prefix='usgs_nwis_0'
        raw_datafiles = [os.path.join(prefix, f) for f in os.listdir(prefix) if f.startswith(file_prefix)]
    return raw_datafiles
