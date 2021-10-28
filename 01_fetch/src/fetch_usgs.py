import os
import urllib

def fetch_params(outfile):
    '''get table of all possible USGS site parameters'''
    params_url = 'https://help.waterdata.usgs.gov/code/parameter_cd_query?fmt=rdb&group_cd=%'
    urllib.request.urlretrieve(params_url, outfile)

def fetch_data(site_list, start_dt, end_dt, outfile):
    '''fetch USGS NWIS data for locations in site_list (gets all parameters available)'''
    for site_num in site_list:
        data_url = f'https://waterservices.usgs.gov/nwis/iv?format=rdb&sites={site_num}&startDT={start_dt}&endDT={end_dt}'
        # download raw data
        outfile_formatted = outfile.format(start_dt=start_dt, end_dt=end_dt, site_num=site_num)
        urllib.request.urlretrieve(data_url, outfile_formatted)

def main():
    site_ids = ['01411390', '01463500', '01464040', '014670261', '01467059', '01467200',
    '01474500', '01474703', '01477050', '01482695', '01482800']

    start_dt = '2019-01-01'
    end_dt = '2019-12-31'

    data_outfile = os.path.join('.', '01_fetch', 'out', '{site_num}_{start_dt}_{end_dt}.txt')
    fetch_data(site_ids, start_dt, end_dt, data_outfile)

    params_outfile = os.path.join('.', '01_fetch', 'out', 'params.txt')
    fetch_params(params_outfile)

if __name__ == '__main__':
    main()