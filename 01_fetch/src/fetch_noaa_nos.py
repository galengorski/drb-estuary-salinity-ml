import pandas as pd
import os
import urllib
import datetime
from dateutil.relativedelta import relativedelta
import requests, json
import yaml
import sys
sys.path.insert(0, os.path.join('01_fetch', 'src'))
import utils

# import config
with open("01_fetch/params_config_fetch_noaa_nos.yaml", 'r') as stream:
    config = yaml.safe_load(stream)

# set up write location data outputs
write_location = config['write_location']
s3_client = utils.prep_write_location(write_location, config['aws_profile'])
s3_bucket = config['s3_bucket']

def fetch_metadata(station_id, metadata_outfile):
    '''fetch tides and currents metadata from NOAA NOS station'''
    metadata_url = f'https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/{station_id}/.json'
    response = requests.get(metadata_url)
    text = response.text
    data = json.loads(text)
    nested_list = pd.json_normalize(data, record_path=['stations'])
    df = nested_list.stack().reset_index()
    metadata = df.iloc[1:, 1:].rename(columns={df.columns[1]: "data_property",
                                               df.columns[2]: (station_id)}).replace(".self", "", regex=True)
    metadata.to_csv(metadata_outfile, index=False)
    if write_location == 'S3':
        print('uploading to s3')
        s3_client.upload_file(metadata_outfile, s3_bucket, local_to_s3_pathname(metadata_outfile))

def fetch_noaa_nos_data(station_id, product, start_year, end_year, datum, time_zone, units, file_format, data_outfile):
    print(f'Fetching {product} data for station {station_id}')
    i=0
    # we can only fetch up to 31 days at a time from this source, so loop through data month by month
    for year in range(int(start_year), int(end_year)+1):
        for month in range(1, 13):
            start_dt = datetime.date(year, month, 1)
            end_dt = start_dt + relativedelta(months=1) - datetime.timedelta(days=1)
            start_dt_str = start_dt.strftime("%Y%m%d")
            end_dt_str = end_dt.strftime("%Y%m%d")
            data_url = f'https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?begin_date=' \
                    f'{start_dt_str}&end_date={end_dt_str}&station={station_id}&product={product}&datum={datum}&time_zone=' \
                    f'{time_zone}&units={units}&format={file_format}'
            data_json = requests.get(data_url).json()
            try:
                # if we are on first iteration for fetching product, overwrite df with data
                if i==0:
                    try:
                        data_df = pd.DataFrame(data_json['data'])
                    except KeyError:
                        data_df = pd.DataFrame(data_json[product])                          
                # otherwise, append the new month of data to the df
                else:
                    try:
                        data_df = pd.concat([data_df,pd.DataFrame(data_json['data'])])
                    except KeyError:
                        data_df = pd.concat([data_df,pd.DataFrame(data_json[product])])
                i+=1
            except:
                continue
    # save dataframe as a csv
    data_outfile_formatted = data_outfile.format(product=product, station_id=station_id)
    try:
        data_df.to_csv(data_outfile_formatted, index=False)
    except:
        data_df = pd.DataFrame()
        data_df.to_csv(data_outfile_formatted, index=False)
    if write_location == 'S3':
        print('uploading to s3')
        s3_client.upload_file(data_outfile_formatted, s3_bucket, local_to_s3_pathname(data_outfile_formatted))

def fetch_site_metadata_file(station_id):
    # site data comes in from snakemake as a set, get single value from set
    if type(station_id)==set:
        station_id = list(station_id)[0]
    metadata_filename = f"noaa_nos_metadata_{station_id}.csv"
    metadata_outfile = os.path.join('.', '01_fetch', 'out', 'metadata', metadata_filename)
    fetch_metadata(station_id, metadata_outfile)

def fetch_single_site_data(station_id, product):
    print(station_id)
    print(product)
    # site data comes in from snakemake as a set, get single value from set
    if type(station_id)==set:
        station_id = list(station_id)[0]
        product = list(product)[0]

    datum = config['datum']
    time_zone = config['time_zone']
    units = config['units']
    file_format = config['file_format']

    start_year = config['start_year']
    end_year = config['end_year']

    filename = "noaa_nos_{station_id}_{product}.csv"
    data_outfile = os.path.join('.', '01_fetch', 'out', filename)
    fetch_noaa_nos_data(station_id, product, start_year, end_year, datum, time_zone, units, file_format, data_outfile)

def fetch_all_sites_data():
    with open("01_fetch/wildcards_fetch_config.yaml", 'r') as stream:
        site_product_config = yaml.safe_load(stream)['fetch_noaa_nos.py']
    site_ids = site_product_config['sites']
    products = site_product_config['products']
    for site_num in site_ids:
        fetch_site_metadata_file(site_num)
        for product in products:
            fetch_single_site_data(site_num, product)

if __name__ == '__main__':
    fetch_all_sites_data()