import sys
import boto3

def prep_write_location(write_location, aws_profile):
    if write_location=='S3':
        cont = input("You are about to write to S3, and you may overwrite existing data. Are you sure you want to do this? (yes, no)")
        if cont=="no":
            sys.exit("Aborting data fetch.")
    # start S3 session so that we can upload data
    session = boto3.Session(profile_name=aws_profile)
    s3_client = session.client('s3')
    return s3_client