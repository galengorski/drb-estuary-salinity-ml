import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dask.array as da
import os
from dask.distributed import Client
import yaml
import utils

client = Client()
client

# import config
with open("01_fetch/params_config_fetch_coawst_model.yaml", 'r') as stream:
    config = yaml.safe_load(stream)

# set up write location data outputs
write_location = config['write_location']
s3_client = utils.prep_write_location(write_location, config['aws_profile'])
s3_bucket = config['s3_bucket']


def load_COAWST_model_run(url):
    '''
    This function is used to read a COAWST model output dataset from a THREDDS url
    and return a data array
    '''
    # load the dataset from the input THREDDS url and chunk it by
    # the ocean_time variable, which measures the time step in the dataset
    # model outputs are on a 3-hour time step
    # A chunk size of 1 was chosen, meaning that data will be split up by timestep.
    ds = xr.open_dataset(url, chunks={'ocean_time':1})
    # read in dataset as an array
    ds = xr.Dataset(ds)
    print(f'Size: {ds.nbytes / (-10**9)} GB')
    return ds

def salt_front_timeseries(ds, river_mile_coords_filepath, run_number):
    # read river mile coordinates csv
    river_mile_coords = pd.read_csv(river_mile_coords_filepath, index_col=0)

    # create array of river miles as points
    target_x = np.array(river_mile_coords.iloc[:,[1]].values).squeeze()
    target_x = xr.DataArray(target_x-1, dims=["dist_miles"]) 
    target_y = np.array(river_mile_coords.iloc[:,[2]].values).squeeze()
    target_y = xr.DataArray(target_y-1,dims=["dist_miles"])
    dist_mile = np.array(river_mile_coords.iloc[:,[0]].values).squeeze()
    dist_mile = xr.DataArray(dist_mile,dims=["points"])

    # sort by river mile, subset values from 1st river mile
    salt = ds.isel(s_rho=0)

    # select variable for timeseries along shore
    salt = salt.isel(xi_rho=target_x,eta_rho=target_y)

    # assign river mile distance as a new coordinate in dataset
    salt = salt.assign_coords({'dist_mile': dist_mile})

    #locate saltfront
    saltfront = salt.where(salt.salt < 0.53).where(salt.salt > 0.5)
    saltfront_location = saltfront.where(saltfront.max('ocean_time'))

    # subset salt variable
    saltfront_location = saltfront_location.salt

    #convert to dataframe
    df = saltfront_location.to_dataframe()

    # tidy dataframe
    df = df[df['salt'].notna()]
    df = df.droplevel(level=1)

    # take daily average
    df = df.resample('1D').mean()

    saltfront_data = os.path.join('.', '01_fetch', 'out', f'salt_front_location_from_COAWST_run_{run_number}.csv')
    df.to_csv(saltfront_data, index=False)
    # upload csv with salt front data to S3
    if write_location == 'S3':
        print('uploading to s3')
        s3_client.upload_file(saltfront_data, s3_bucket, '01_fetch/out/'+os.path.basename(saltfront_data))

def main():
    # define model run
    url = config['url']
    u = url.split('/')
    run_number = u[12]

    # define csv with river mile coordinates
    river_mile_coords_filepath = config['river_mile_coords_filepath']

    ds = load_COAWST_model_run(url)
    salt_front_timeseries(ds, river_mile_coords_filepath, run_number)

if __name__ == '__main__':
    main()
