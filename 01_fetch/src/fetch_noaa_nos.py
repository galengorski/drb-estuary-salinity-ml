import pandas as pd
import os 
import numpy as np
import urllib
import requests, json

def fetch_metadata(station_id, metadata_outfile, bucket, write_location):
    '''fetch tides and currents metadata from NOAA NOS station'''
    metadata_url = f'https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/{station_id}/.json'
    response = requests.get(metadata_url)
    text = response.text
    data = json.loads(text)
    nested_list = pd.json_normalize(data, record_path=['stations'])
    df = nested_list.stack().reset_index()
    metadata = df.iloc[1:,1:].rename(columns={df.columns[1]:"data_property",
                                              df.columns[2]:(station_id)}).replace(".self","", regex=True)
    metadata.to_csv(metadata_outfile, index=False)
    if write_location == 'S3':
        print('uploading to s3')
        s3_client.upload_file(metadata_outfile, bucket, '01_fetch/out/'+os.path.basename(metadata_outfile))

def fetch_noaa_nos_data(start_dt, end_dt, datum, station_id, time_zone, product, units, file_format, data_outfile, bucket, write_location):
    '''fetch NOAA NOS data from select station. Change product argument for NOAA NOS data product. (ie. product = current for current data)'''
        data_url = f'"https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?product=predictions&application=NOS.COOPS.TAC.WL&begin_date={start_dt}&end_date={end_dt}&datum={datum}&station={station_id}&product={product}&time_zone={time_zone}&units={units}&interval=&format={file_format}"
    urllib.request.urlretrieve(data_url, data_outfile)
    if write_location == 'S3':
        print('uploading to s3')
        s3_client.upload_file(data_outfile, bucket, '01_fetch/out/'+os.path.basename(data_outfile))

def main():
    '''choose where you want to write your data outputs: local or S3'''
    write_location: 'local'
    '''set name of AWS profile storing credentials for S3'''
    aws_profile: 'dev'
    '''set AWS bucket to read/write to'''
    s3_bucket: 'drb-estuary-salinity'
    '''set up AWS client'''
    s3_client = utils.prep_write_location(write_location, aws_profile)
  
    station_id = '8551762'
    
    datum='MLLW'
    ''' options:
    HAT: Highest Astronomical Tide
    MHHW: Mean Higher High Water
    MHW: Mean High Water
    DTL: Diurnal Tide Level
    MTL: Mean Tide Level
    MSL: Mean Sea Level
    MLW: Mean Low Water 
    MLLW: Mean Lower Low Water
    LAT: Lowest Astronomical Tide
    GT: Great Diurnal Range
    MN: Mean Diurnal Range
    DHQ: Mean Diurnal High Water Inequality
    HWI: Greenwich High Water Interval
    LWI: Greenwich Low Water Interval
    Max Tide: Highest Observed Tide
    Min Tide: Lowest Observed Tide
    Station Datum: fixed base elevation at a tide station
    National Tidal Datum Epoch: The specific 19-year period adopted by the National Ocean Service as the official time segment over which tide observations are taken and reduced to obtain mean values (e.g., mean lower low water, etc.) for tidal datums.'''
    
    product = 'water_level'
    '''options: water_level, air_temperature, water_temperature, wind, air_pressure, air_gap, conductivity, visibility, humidity, salinity, hourly_height,
    high_low, daily_mean, monthly_mean, one_minute_water_level, predictions, datums, currents, and currents_predictions.'''
    
    time_zone='GMT'
    '''options:
    gmt: Greenwich Mean Time
    lst: Local Standard Time. The time local to the requested station.
    lst_ldt: Local Standard/Local Daylight Time. The time local to the requested station.'''
    
    units='metric'
    '''options: english and metric'''
    
    file_format='csv'

    start_dt = '20190101'
    end_dt = '20191231'

    data_outfile = os.path.join("fetch","out", f"noaa_nos_{product}_{station_id}.csv")
    fetch_noaa_nerrs_data(start_dt, end_dt, datum, station_id, time_zone, product, units, file_format, data_outfile)

    metadata_outfile = os.path.join("fetch","out", "metadata.csv")
    fetch_metadata(station_id, metadata_outfile)

if __name__ == '__main__':
    main()
