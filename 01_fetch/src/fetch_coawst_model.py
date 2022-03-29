import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dask.array as da
import os
from dask.distributed import Client
client = Client()
client

def fetch_COAWST_model_run(url):
    ds = xr.open_dataset(url, chunks={'ocean_time':720})
    ds = xr.Dataset(ds, coords={'lon': (['eta_rho', 'xi_rho'], nc['lon_rho']),
                          'lat': (['eta_rho', 'xi_rho'], nc['lat_rho']),
                          's': nc['s_rho'])
    print(f'Size: {ds.nbytes / (-10**9)} GB')
    u = url.split('/')
    print(u[12])
    return ds
                                
def river_mile_timeseries():
    # read river mile coordinates csv
    river_mile_coords = pd.read_csv(river_mile_coords_filepath, index_col=0)
    
    # create array of river miles as points
    target_x = np.array(river_mile_coords.iloc[:,[1]].values).squeeze()
    target_x = xr.DataArray(target_x,dims=["points"]) 
    target_y = np.array(river_mile_coords.iloc[:,[2]].values).squeeze()
    target_y = xr.DataArray(target_y,dims=["points"]) 
    dist_mile = np.array(river_mile_coords.iloc[:,[0]].values).squeeze()
    dist_mile = xr.DataArray(dist_mile,dims=["points"]) 
    
    # select variable for timeseries along shore
    ds = ds[variable].isel(xi_rho=ds.lon,eta_rho=ds.lat) 
    
    # assign river mile distance as a new coordinate in dataset
    ds = ds.assign_coords({'dist_mile': dist_mile})
    
    # sort by river mile, subset values from 1st river mile
    salt = ds.sortby(ds.dist_mile).isel(s_rho=0).where(salt>0.5)
    
    # select salt variable
    salt = salt.salt
    print(f'Size: {salt.nbytes / (-10**9)} GB')
    
    # create new netcdf with subsetted data
    if write_location == 'S3':
        try: 
            s3.Object(s3_bucket, netcdf_path).load()
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                salt.to_netcdf(netcdf_path)
                print('uploading to s3')
                s3_client.upload_file(s3_bucket, '01_fetch/out/' + netcdf_filename)
    else:
        if os.path.isfile(netcdf_path):
            print ("File exist")
        else:
        salt.to_netcdf(netcdf_path)
        
    # close original salt dataset; improves performance
    salt.close()
                                
def main():
    # import config
    with open("01_fetch/fetch_config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)['fetch_COAWST_model_run.py']
        
    # set up write location data outputs
    write_location = config['write_location']
    s3_client = utils.prep_write_location(write_location, config['aws_profile'])
    s3_bucket = config['s3_bucket']
        
    url = config['url']
    
    river_mile_coords_filepath = config['river_mile_coords_filepath']
    variable = config['variable']
    
    netcdf_filename = f"coawst_salt_{model_run}_{model_run_year}.nc"
    netcdf_path = os.path.join('.', '01_fetch', 'out', filename)
if __name__ == '__main__':
    main()