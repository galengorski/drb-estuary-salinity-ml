import os
import urllib
import boto3

def fetch_params(outfile, s3_client):
    '''get table of all possible USGS site parameters'''
    params_url = 'https://help.waterdata.usgs.gov/code/parameter_cd_query?fmt=rdb&group_cd=%'
    print('fetcing parameter file and saving locally')
    urllib.request.urlretrieve(params_url, outfile)
    print('uploading to s3')
    s3_client.upload_file(outfile, 'drb-estuary-salinity', '01_fetch_out/'+os.path.basename(outfile))

def fetch_data(site_list, start_dt, end_dt, outfile, s3_client):
    '''fetch USGS NWIS data for locations in site_list (gets all parameters available)'''
    for site_num in site_list:
        data_url = f'https://waterservices.usgs.gov/nwis/iv?format=rdb&sites={site_num}&startDT={start_dt}&endDT={end_dt}'
        outfile_formatted = outfile.format(start_dt=start_dt, end_dt=end_dt, site_num=site_num)
        print(f'fetcing data for site {site_num} and saving locally')
        urllib.request.urlretrieve(data_url, outfile_formatted)
        print('uploading to s3')
        s3_client.upload_file(outfile_formatted, 'drb-estuary-salinity', '01_fetch_out/'+os.path.basename(outfile_formatted))

def main():
    # start S3 session so that we can upload data
    session = boto3.Session(profile_name='dev')
    s3_client = session.client('s3')

    site_ids = ['01411390', '01463500', '01464040', '014670261', '01467059', '01467200',
    '01474500', '01474703', '01477050', '01482695', '01482800']
    start_dt = '2019-01-01'
    end_dt = '2019-12-31'

    # fetch raw data files
    data_outfile_txt = os.path.join('.', '01_fetch', 'out', '{site_num}_{start_dt}_{end_dt}.txt')
    fetch_data(site_ids, start_dt, end_dt, data_outfile_txt, s3_client)

    # fetch parameter file
    params_outfile_txt = os.path.join('.', '01_fetch', 'out', 'params.txt')
    fetch_params(params_outfile_txt, s3_client)

if __name__ == '__main__':
    main()