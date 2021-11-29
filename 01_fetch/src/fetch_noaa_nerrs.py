import pandas as pd
import os 
import numpy as np
import urllib
import requests, json

def fetch_metadata(station_id, metadata_outfile):
    '''fetch metadata for NOAA NERRS station'''
    metadata_url = f'https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/{station_id}/.json'
    response = requests.get(metadata_url)
    text = response.text
    data = json.loads(text)
    nested_list = pd.json_normalize(data, record_path=['stations'])
    df = nested_list.stack().reset_index()
    metadata = df.iloc[1:,1:].rename(columns={df.columns[1]:"NOAA NERRS metadata:",
                                              df.columns[2]:(station_id)}).replace(".self","", regex=True)
    metadata.to_csv(metadata_outfile, index=False)

def fetch_noaa_nerrs_data(start_dt, end_dt, datum, station_id, time_zone, product, units, file_format, data_outfile):
    '''fetch NOAA NERRS data from select station. Change product argument for NOAA data product. (ie. product = current for current data)'''
    dates = pd.date_range(start=start_dt, end=end_dt, freq='D')
    dates = dates.strftime('%Y%m%d').astype(int)
    url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?product=predictions&application=NOS.COOPS.TAC.WL"
    data_url = f'{url}&begin_date={dates[0]}&end_date={dates[-1]}&datum={datum}&station={station_id}&product={product}&time_zone={time_zone}&units={units}&interval=&format={file_format}'
    outfile_format = data_outfile.format(start_dt=start_dt, end_dt=end_dt, station_id=station_id, data_outfile=data_outfile, product=product)
    urllib.request.urlretrieve(data_url, outfile_format)

def main():
    station_id = '8551762'
    
    datum='MLLW'
    
    product = 'water_level'
    
    time_zone='GMT'
    
    units='english'
    
    file_format='csv'

    start_dt = '2019-01-01'
    end_dt = '2019-12-31'

    data_outfile = os.path.join("fetch","out", "{product}_{station_id}_{start_dt}_{end_dt}.csv")
    fetch_noaa_nerrs_data(start_dt, end_dt, datum, station_id, time_zone, product, units, file_format, data_outfile)

    metadata_outfile = os.path.join("fetch","out", "metadata.csv")
    fetch_metadata(station_id, metadata_outfile)

if __name__ == '__main__':
    main()