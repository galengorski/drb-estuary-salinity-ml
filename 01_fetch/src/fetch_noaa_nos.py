import pandas as pd
import os
import urllib
import requests, json
import utils

def fetch_metadata(station_id, metadata_outfile, bucket, write_location, s3_client):
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
        s3_client.upload_file(metadata_outfile, bucket, '01_fetch/out/'+os.path.basename(metadata_outfile))

def fetch_noaa_nos_data(start_dt, end_dt, datum, station_id, time_zone, product, units, file_format, data_outfile, bucket, write_location, s3_client):
    for products in product:
        data_url = f'https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?begin_date=' \
                   f'{start_dt}&end_date={end_dt}&datum={datum}&station={station_id}&product={products}&time_zone=' \
                   f'{time_zone}&units={units}&interval=&format={file_format}'
        data_outfile_formatted = data_outfile.format(products=products, station_id=station_id)
        urllib.request.urlretrieve(data_url, data_outfile_formatted)
        if write_location == 'S3':
            print('uploading to s3')
            s3_client.upload_file(data_outfile_formatted, bucket, '01_fetch/out/'+os.path.basename(data_outfile_formatted))


def main():
    # import config
    with open("01_fetch/fetch_config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)['fetch_noaa_nos.py']
        
    #choose where you want to write location for data outputs
    write_location = config['write_location']
    s3_client = utils.prep_write_location(write_location, config['aws_profile'])
    s3_bucket = config['s3_bucket']

    station_id = config['station_id']

    datum = config['datum']

    product = config['product']

    time_zone = config['time_zone']

    units = config['units']
    # options: english and metric

    file_format = config['file_format']

    start_dt = config['start_dt']
    end_dt = config['end_dt']

    path = os.path.dirname('/01_fetch/out/')
    os.path.isdir(path)
    filename = "noaa_nos_{products}_{station_id}.csv"
    data_outfile = os.path.join('.', path + '/' + filename)
    fetch_noaa_nos_data(start_dt, end_dt, datum, station_id, time_zone, product, units, file_format, data_outfile, s3_bucket, write_location, s3_client)

    path = os.path.dirname('/01_fetch/out/')
    metadata_filename = f"noaa_nos_metadata_{products}_{station_id}.csv"
    metadata_outfile = os.path.join('.', path + '/' + metadata_filename)
    fetch_metadata(station_id, metadata_outfile, s3_bucket, write_location, s3_client)


if __name__ == '__main__':
    main()
