import requests
from bs4 import BeautifulSoup
import pandas as pd

URL = "https://www.ncaa.com/stats/baseball/d1/current/individual/200"
response = requests.get(URL)
soup = BeautifulSoup(response.content, 'html.parser')

columns = ['Rank', 'Name', 'Team', 'CL', 'Position', 'G', 'AB', 'H', 'BA']

df = pd.DataFrame(columns=columns)

table = soup.find("table", attrs = {'class':'block-stats__stats-table',
                                    'data-striping':'1'}).tbody

trs = table.find_all('tr')

for tr in trs:
    tds  = tr.find_all('td')
    row = [td.text.replace('\n', '') for td in tds]
    df = df.append(pd.Series(row, index = columns), ignore_index = True)

df.to_csv('Batting Average Leaders.csv', index=False)

