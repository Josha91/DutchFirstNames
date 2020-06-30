"""
Get the top ~100 (boys and girls) in baby names in the Netherlands from 1880-2014, 
write out to .csv file

Data from Meertens institute Voornamenbank
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import IPython
import numpy as np
import matplotlib.pyplot as plt 
from selenium.webdriver import Chrome, Firefox
import time

def _get_table(soup,tag):
	table = {'rank':[],'naam':[],'aantal':[],'tag':[]}
	for tmp in soup.find_all('tr')[1:]:
		data = tmp.find_all('td')
		table['rank'].append(int(data[0].text))
		table['naam'].append(data[1].text)
		table['aantal'].append(int(data[2].text))
		table['tag'].append(tag)
	return pd.DataFrame(table)

def download_all():
	link = lambda x: 'https://www.meertens.knaw.nl/nvb/topnamen/land/Nederland/%d'%x
	table_all = pd.DataFrame({'rank':[],'naam':[],'aantal':[],'tag':[]})
	for yr in np.arange(1880,2015):
		print(yr)
		result = requests.get(link(yr))
		src = result.text
		soup = BeautifulSoup(src,'lxml')

		jongens = soup.find('table',id='topnamen-jongens')
		table_jongens = _get_table(jongens,'M')
		
		meisjes = soup.find('table',id='topnamen-meisjes')
		table_meisjes = _get_table(meisjes,'F')

		table_all = pd.concat((table_all,table_jongens,table_meisjes))
	return table_all


if __name__ == '__main__':
	t = download_all()
	t.to_csv('name_popularity_netherlands_1880-2014.csv')

