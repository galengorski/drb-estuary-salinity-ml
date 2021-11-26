import os
import urllib
from suds.client import Client
from lxml import etree


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
    soapClient = Client("http://cdmo.baruch.sc.edu/webservices2/requests.cfc?wsdl", timeout=90, retxml=True)
    site_ids = ['DELSJMET']
    start_dt = '2019-01-01'
    end_dt = '2019-12-31'

    data_outfile_txt = os.path.join('.', '01_fetch', 'out', '{site_num}_{start_dt}_{end_dt}.txt')
    fetch_data(site_ids, start_dt, end_dt, data_outfile_txt)

    params_outfile_txt = os.path.join('.', '01_fetch', 'out', 'params.txt')
    fetch_params(params_outfile_txt)

if __name__ == '__main__':
    main()




soapClient = Client("http://cdmo.baruch.sc.edu/webservices2/requests.cfc?wsdl", timeout=90, retxml=True)

#Get the station codes SOAP request example.
station_codes = soapClient.service.exportStationCodesXML()
display(station_codes)
doc = etree.XML(station_codes)
print(doc.findtext('code'))


#Get all parameters from the station NIWOLMET for the date range of 2014-12-30 to 2014-12-31
params = soapClient.service.exportAllParamsDateRangeXML('delsjmet', '2019-01-01', '2019-12-31', '*')
print(params)

with open('.//01_fetch//out//test.txt', 'w') as f:
    f.write(params.decode('utf-8'))