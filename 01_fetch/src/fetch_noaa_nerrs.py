import os
import yaml
import glob
from zipfile import ZipFile
import shutil
import sys
sys.path.insert(0, os.path.join('01_fetch', 'src'))
import utils


# import config
with open("01_fetch/params_config_fetch_noaa_nerrs.yaml", 'r') as stream:
    config = yaml.safe_load(stream)

# set up write location data outputs
write_location = config['write_location']
s3_client = utils.prep_write_location(write_location, config['aws_profile'])
s3_bucket = config['s3_bucket']

def fetch_single_site_data(station_id, zipfile_path):
    # site data comes in from snakemake as a set, get single value from set
    if type(station_id)==set:
        station_id = list(station_id)[0]

    # unzip data folder
    unzipped_folder = os.path.splitext(zipfile_path)[0]
    with ZipFile(zipfile_path, 'r') as zip_ref:
        zip_ref.extractall(unzipped_folder)
    
    # get data files of interest
    file_list = glob.glob(os.path.join(f'{unzipped_folder}', f'{station_id}*.csv'))

    # rename and move raw data files to out
    for f in file_list:
        year = f[-8:-4]
        data_outfile = os.path.join('.', '01_fetch', 'out', f'noaa_nerrs_{station_id}_{year}.csv')
        shutil.move(f,data_outfile)
        if write_location == 'S3':
            print('uploading to s3')
            s3_client.upload_file(metadata_outfile, s3_bucket, local_to_s3_pathname(metadata_outfile))

    # delete all unzipped contents left
    shutil.rmtree(unzipped_folder) 

def fetch_all_sites_data():
    with open("01_fetch/wildcards_fetch_config.yaml", 'r') as stream:
        site_config = yaml.safe_load(stream)['fetch_noaa_nerrs.py']
    site_ids = site_config['sites']
    zip_folders = config['zip_folders']
    for station_id in site_ids:
        zipfilename = '{}.zip'.format(zip_folders[station_id])
        zipfile_path = os.path.join('01_fetch', 'in', zipfilename)
        # download zip file of data from s3 if we don't have it
        if not os.path.exists(zipfile_path):
            utils.download_s3_to_local(os.path.dirname(zipfile_path).replace('\\', '/'), os.path.dirname(zipfile_path), zipfilename)
        # get data for single site from zipfile
        fetch_single_site_data(station_id, zipfile_path)

if __name__ == '__main__':
    fetch_all_sites_data()