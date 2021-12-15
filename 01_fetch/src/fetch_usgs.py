import os
import urllib
import yaml
import utils

def fetch_params(outfile, bucket, write_location, s3_client):
    '''get table of all possible USGS site parameters'''
    params_url = 'https://help.waterdata.usgs.gov/code/parameter_cd_query?fmt=rdb&group_cd=%'
    print('fetcing parameter file and saving locally')
    urllib.request.urlretrieve(params_url, outfile)
    if write_location == 'S3':
        print('uploading to s3')
        s3_client.upload_file(outfile, bucket, '01_fetch/out/'+os.path.basename(outfile))

def fetch_data(site_num, start_dt, end_dt, outfile, bucket, write_location, s3_client):
    '''fetch USGS NWIS data for locations in site_list (gets all parameters available)'''
    data_url = f'https://waterservices.usgs.gov/nwis/iv?format=rdb&sites={site_num}&startDT={start_dt}&endDT={end_dt}'
    print(f'fetcing data for site {site_num} and saving locally')
    urllib.request.urlretrieve(data_url, outfile)
    if write_location == 'S3':
        print('uploading to s3')
        s3_client.upload_file(outfile, bucket, '01_fetch/out/'+os.path.basename(outfile))

def main():
    # import config
    with open("01_fetch/fetch_config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)['fetch_usgs.py']

    # set up write location data outputs
    write_location = config['write_location']
    s3_client = utils.prep_write_location(write_location, config['aws_profile'])
    s3_bucket = config['s3_bucket']

    # usgs nwis sites we want to fetch data for
    site_ids = config['site_ids']

    # start and end dates for data fetch
    start_dt = config['start_dt']
    end_dt = config['end_dt']

    # fetch raw data files
    for site_num in site_ids:
        data_outfile_txt = os.path.join('.', '01_fetch', 'out', f'usgs_nwis_{site_num}.txt')
        fetch_data(site_num, start_dt, end_dt, data_outfile_txt, s3_bucket, write_location, s3_client)

    # fetch parameter file
    params_outfile_txt = os.path.join('.', '01_fetch', 'out', 'usgs_nwis_params.txt')
    fetch_params(params_outfile_txt, s3_bucket, write_location, s3_client)

if __name__ == '__main__':
    main()