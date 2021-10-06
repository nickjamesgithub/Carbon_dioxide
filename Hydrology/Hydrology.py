import requests
# url
url = 'http://www.bom.gov.au/water/hrs/content/data/410713/410713_daily_ts.csv'
r = requests.get(url, headers={'User-agent': 'Mozilla/5.0'})
open('test_download1.csv', 'wb').write(r.content)