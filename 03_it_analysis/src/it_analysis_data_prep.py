# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 10:17:00 2021

@author: ggorski
"""

import boto3
import pandas as pd
import io
import subprocess
import os

#%%
def download_s3_to_local(s3_dir_prefix, local_outdir, file_id):
    '''download data files from s3 bucket to local machine for
    development
    file_id - a file identifier substring that is contained within all 
    the file names you want to download. For example 'usgs_nwis' will 
    download all files with 'usgs_nwis' in the file name'''
    
    # after you have configured saml2aws, you can log in and create a new 
    # token by executing the following command:
    subprocess.run(["saml2aws", "login", "--skip-prompt", "--role", "arn:aws:iam::807615458658:role/adfs-wma-developer"])
    
    # read in S3 credentials from ./.aws/credentials file
    # assumes we are using a credential profile names 'dev'
    session = boto3.Session(profile_name='dev')
    s3_client = session.client('s3')
    # end the name of the bucket you want to read/write to:
    s3_bucket = 'drb-estuary-salinity'
    
    # define the local file path that you want to save this file to
    os.makedirs(local_outdir, exist_ok=True)
    
    
    # define a file prefix to look in your bucket for

    # loop through all objects with this prefix and print
    for obj in s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=s3_dir_prefix)['Contents']:
        s3_fpath = obj['Key']
        if ".csv" and file_id not in s3_fpath:
            continue
        local_fpath = os.path.join(local_outdir,obj['Key'].split('/')[2])
        s3_client.download_file(s3_bucket, s3_fpath, local_fpath)
        print(s3_fpath+' Downloaded to local')
    
