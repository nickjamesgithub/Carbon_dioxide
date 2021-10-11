import requests
import pandas as pd
import os

# Read in stations data
stations = pd.read_csv("/Users/tassjames/Desktop/hydrology/hrs_station_details.csv", skiprows = 11)
station_number = stations["AWRC Station Number"]

for i in range(len(station_number)):
    # Station ticker
    station_ticker = station_number[i]

    # url
    url = 'http://www.bom.gov.au/water/hrs/content/data/'+str(station_ticker)+'/'+str(station_ticker)+'_daily_ts.csv'
    r = requests.get(url, headers={'User-agent': 'Mozilla/5.0'})
    open(os.path.join('/Users/tassjames/Desktop/hydrology/data/')+station_ticker+'.csv', 'wb').write(r.content)
    print(i)