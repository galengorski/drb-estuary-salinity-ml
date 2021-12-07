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
    metadata = df.iloc[1:,1:].rename(columns={df.columns[1]:"data_property",
                                              df.columns[2]:(station_id)}).replace(".self","", regex=True)
    metadata.to_csv(metadata_outfile, index=False)

def fetch_noaa_nerrs_data(start_dt, end_dt, datum, station_id, time_zone, product, units, file_format, data_outfile):
    '''fetch NOAA NERRS data from select station. Change product argument for NOAA data product. (ie. product = current for current data)'''
        data_url = f'"https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?product=predictions&application=NOS.COOPS.TAC.WL&begin_date={start_dt}&end_date={end_dt}&datum={datum}&station={station_id}&product={product}&time_zone={time_zone}&units={units}&interval=&format={file_format}"
    urllib.request.urlretrieve(data_url, data_outfile)

def main():
    station_id = '8551762'
    
    datum='MLLW'
    
    product = 'water_level'
    
    time_zone='GMT'
    
    units='english'
    
    file_format='csv'

    start_dt = '20190101'
    end_dt = '20191231'

    data_outfile = os.path.join("fetch","out", f"noaa_{product}_{station_id}.csv")
    fetch_noaa_nerrs_data(start_dt, end_dt, datum, station_id, time_zone, product, units, file_format, data_outfile)

    metadata_outfile = os.path.join("fetch","out", "metadata.csv")
    fetch_metadata(station_id, metadata_outfile)

if __name__ == '__main__':
    main()