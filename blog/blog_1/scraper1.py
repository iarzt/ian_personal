from bs4 import BeautifulSoup
import pandas as pd
import requests
import os

def scrape(URL, name):

    try:
        response = requests.request('GET', URL)

        soup = BeautifulSoup(response.text.encode('utf8'), 'html.parser')

        columns = ['Year', 'Pitch Type', '#', '# RHB', '# LHB', '%', 'MPH', 'PA', 'AB', 'H', '1B', '2B',
                    '3B', 'HR', 'SO', 'BBE', 'BA', 'XBA', 'SLG', 'XSLG', 'WOBA', 'XWOBA', 'EV', 'LA', 'Spin',
                    'Whiff %', 'PutAway %']

        df = pd.DataFrame(columns=columns)

        table = soup.find("table", attrs = {'id':'detailedPitches'}).tbody
        trs = table.find_all('tr')

        for tr in trs:
            tds  = tr.find_all('td')
            row = [td.text.replace('\n', '') for td in tds]
            df = df.append(pd.Series(row, index = columns), ignore_index = True)

        new_name = ''
        for i in range(len(name)):
            if name[i].isalpha() == True:
                new_name += name[i]
            else:
                new_name += ' '

        df[''] = new_name

        print(new_name, "SAVED")

        return df

    except:
        pass


def main():

    name_list = []

    in_file = open("./names.txt", "r")

    for line in in_file:
        line = line.strip('\n')
        name_list.append(line)

    in_file.close()

    f = 'active_spin_examples.csv'

    for name in name_list:
        URL = "https://baseballsavant.mlb.com/savant-player/" + name + "?stats=statcast-r-pitching-mlb"
        df = scrape(URL, name)
        if not os.path.isfile(f):
           df.to_csv(f, header='column_names')
        else:
           df.to_csv(f, mode='a', header=False)

if __name__ == '__main__':
    main()
