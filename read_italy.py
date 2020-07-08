import requests
from bs4 import BeautifulSoup
import pandas as pd
import IPython
import numpy as np
import matplotlib.pyplot as plt 
from selenium.webdriver import Chrome, Firefox
from selenium.webdriver.support.ui import Select
import unidecode
import time

def get_table(soup,year):
	boys = soup.find_all('td',class_='odd male text')+soup.find_all('td',class_='even male text')
	girls = soup.find_all('td',class_='odd female text')+soup.find_all('td',class_='even female text')

	df = {'name':[],'frequency':[],'percentage':[],'ranking':[],'gender':[],'year':[]}
	for obj in boys:
		df['name'].append(obj.text)
		df['frequency'].append(int(obj.next_sibling.text.replace('.','')))
		df['percentage'].append(float(obj.next_sibling.next_sibling.text.replace(',','.')))
		df['ranking'].append(int(obj.previous))
		df['gender'].append("M")
		df['year'].append(year)

	for obj in girls:
		df['name'].append(obj.text)
		df['frequency'].append(int(obj.next_sibling.text.replace('.','')))
		df['percentage'].append(float(obj.next_sibling.next_sibling.text.replace(',','.')))
		df['ranking'].append(int(obj.find_previous('td',class_='odd male text').previous))
		df['gender'].append('F')
		df['year'].append(year)

	return pd.DataFrame(df).sort_values('ranking')

def vowel_count(string):
	lowercase = string.lower()
	lowercase = unidecode.unidecode(lowercase)
	count = 0
	for vowel in 'aeiou':
		count += lowercase.count(vowel)
	return count

def read_italy(browser=None):
	db = 'https://www.istat.it/it/dati-analisi-e-prodotti/contenuti-interattivi/contanomi'
	if browser is None:
		browser = Firefox(executable_path='/Users/houdt/Downloads/geckodriver')

	browser.get(db) #open the database
	time.sleep(2) #time to load

	#Available years
	years = browser.find_element_by_id('year').text.split('\n') 
	#years = [int(yr) for yr in years]

	df = pd.DataFrame({})
	for yr in years:

		#Select year...
		select = Select(browser.find_element_by_id('year'))
		#Select number of results per page... 
		select2 = Select(browser.find_element_by_id('top'))
		select.select_by_visible_text(yr)
		select2.select_by_visible_text('50')

		browser.find_element_by_id('sendTop').click()
		time.sleep(0.5)

		html = browser.page_source
		soup = BeautifulSoup(html,'html.parser')
		table = get_table(soup,int(yr))
		df = df.append(table)

	browser.quit()
	df = df.sort_values('year')
	df.to_csv('name_statistics_italy_%d-%d.csv'%(df['year'].min(),df['year'].max()),index_label=False)

def plot_average_length(ax=None):
	df = pd.read_csv('name_statistics_italy_1999-2018.csv')
	df['Length'] = df['name'].apply(lambda x: len(x))

	#group = cat.loc[(cat.gender==gender)&(cat.name.isin(names))].groupby('name',sort=False)
	#x = group.apply(lambda x: x['median age'].unique()[0])

	if ax is None: fig,ax = plt.subplots(1)
	group = df.loc[df['gender']=='F'].groupby('year')
	y = group.apply(lambda x: x['Length'].mean())
	x = group.apply(lambda x: x['year'].mean())
	ax.plot(x,y,color='blue',label='Italy')

	group = df.loc[df['gender']=='M'].groupby('year')
	y = group.apply(lambda x: x['Length'].mean())
	x = group.apply(lambda x: x['year'].mean())
	ax.plot(x,y,color='blue',linestyle='dashed')
	ax.set_xlim([1920,2020])
	plot_average_length_NL(ax)
	plot_average_length_belgium(ax)
	plot_average_length_UK(ax)
	plot_average_length_usa(ax)
	ax.set_ylabel('Average name length [characters]',fontsize=16)
	ax.set_xlabel("Year",fontsize=16)
	ax.legend()
	if ax is None: plt.show()

def plot_average_length_NL(ax=None):
	df = pd.read_csv('name_statistics_netherlands_1880-2014.csv')
	df['Length'] = df['name'].apply(lambda x: len(x))

	#group = cat.loc[(cat.gender==gender)&(cat.name.isin(names))].groupby('name',sort=False)
	#x = group.apply(lambda x: x['median age'].unique()[0])

	if ax is None: fig,ax = plt.subplots(1)
	group = df.loc[df['gender']=='vrouw'].groupby('year')
	y = group.apply(lambda x: x['Length'].mean())
	x = group.apply(lambda x: x['year'].mean())
	ax.plot(x,y,color='orange',linestyle='solid',label='the Netherlands')

	group = df.loc[df['gender']=='man'].groupby('year')
	y = group.apply(lambda x: x['Length'].mean())
	x = group.apply(lambda x: x['year'].mean())
	ax.plot(x,y,color='orange',linestyle='dashed')
	if ax is None: plt.show()

def plot_average_length_belgium(ax=None):
	df = pd.read_csv('name_statistics_belgium_1995-2017.csv')
	df['Length'] = df['name'].apply(lambda x: len(x))

	#group = cat.loc[(cat.gender==gender)&(cat.name.isin(names))].groupby('name',sort=False)
	#x = group.apply(lambda x: x['median age'].unique()[0])

	if ax is None: fig,ax = plt.subplots(1)
	group = df.loc[df['gender']=='F'].groupby('year')
	y = group.apply(lambda x: x['Length'].mean())
	x = group.apply(lambda x: x['year'].mean())
	ax.plot(x,y,color='k',linestyle='solid',label='Belgium')

	group = df.loc[df['gender']=='M'].groupby('year')
	y = group.apply(lambda x: x['Length'].mean())
	x = group.apply(lambda x: x['year'].mean())
	ax.plot(x,y,color='k',linestyle='dashed')
	if ax is None: plt.show()

def plot_average_length_usa(ax=None):
	df = pd.read_csv('name_statistics_USA.csv')
	df['Length'] = df['name'].apply(lambda x: len(x))

	#group = cat.loc[(cat.gender==gender)&(cat.name.isin(names))].groupby('name',sort=False)
	#x = group.apply(lambda x: x['median age'].unique()[0])

	if ax is None: fig,ax = plt.subplots(1)
	group = df.loc[df['gender']=='F'].groupby('year')
	y = group.apply(lambda x: x['Length'].mean())
	x = group.apply(lambda x: x['year'].mean())
	ax.plot(x,y,color='green',linestyle='solid',label='USA')

	group = df.loc[df['gender']=='M'].groupby('year')
	y = group.apply(lambda x: x['Length'].mean())
	x = group.apply(lambda x: x['year'].mean())
	ax.plot(x,y,color='green',linestyle='dashed')
	if ax is None: plt.show()

def plot_average_length_UK(ax=None):
	df = pd.read_csv('name_statistics_UK_1996-2019.csv')
	df['Length'] = df['Name'].apply(lambda x: len(x))

	#group = cat.loc[(cat.gender==gender)&(cat.name.isin(names))].groupby('name',sort=False)
	#x = group.apply(lambda x: x['median age'].unique()[0])

	if ax is None: fig,ax = plt.subplots(1)
	group = df.loc[df['Gender']=='F'].groupby('Year')
	y = group.apply(lambda x: x['Length'].mean())
	x = group.apply(lambda x: x['Year'].mean())
	ax.plot(x,y,color='red',linestyle='solid',label='UK')

	group = df.loc[df['Gender']=='M'].groupby('Year')
	y = group.apply(lambda x: x['Length'].mean())
	x = group.apply(lambda x: x['Year'].mean())
	ax.plot(x,y,color='red',linestyle='dashed')
	if ax is None: plt.show()


	#Figure out: how to find the years without going through the overbodige apply in line 79
	#How to plot directly in seaborn from DataFrame?

def plot_vowel_fraction(ax=None):
	df = pd.read_csv('name_statistics_italy_1999-2018.csv')
	df['Vowels'] = df['name'].apply(lambda x: vowel_count(x)/len(x))

	if ax is None: fig,ax = plt.subplots(1)
	group = df.loc[df['gender']=='F'].groupby('year')
	y = group.apply(lambda x: x['Vowels'].mean())
	x = group.apply(lambda x: x['year'].mean())
	ax.plot(x,y,color='blue',label='Women')

	group = df.loc[df['gender']=='M'].groupby('year')
	y = group.apply(lambda x: x['Vowels'].mean())
	x = group.apply(lambda x: x['year'].mean())
	ax.plot(x,y,color='blue',label='Men',linestyle='dashed')
	ax.legend()
	if ax is None: plt.show()

def plot_vowel_fraction_UK(ax=None):
	df = pd.read_csv('name_statistics_UK_1996-2019.csv')
	df['Vowels'] = df['Name'].apply(lambda x: vowel_count(x)/len(x))

	if ax is None: fig,ax = plt.subplots(1)
	group = df.loc[df['Gender']=='F'].groupby('Year')
	y = group.apply(lambda x: x['Vowels'].mean())
	x = group.apply(lambda x: x['Year'].mean())
	ax.plot(x,y,color='red')

	group = df.loc[df['Gender']=='M'].groupby('Year')
	y = group.apply(lambda x: x['Vowels'].mean())
	x = group.apply(lambda x: x['Year'].mean())
	ax.plot(x,y,color='r',linestyle='dashed')
	if ax is None: plt.show()

def plot_vowel_fraction_belgium(ax=None):
	df = pd.read_csv('name_statistics_belgium_1995-2017.csv')
	df['Vowels'] = df['name'].apply(lambda x: vowel_count(x)/len(x))

	if ax is None: fig,ax = plt.subplots(1)
	group = df.loc[df['gender']=='F'].groupby('year')
	y = group.apply(lambda x: x['Vowels'].mean())
	x = group.apply(lambda x: x['year'].mean())
	ax.plot(x,y,color='k')

	group = df.loc[df['gender']=='M'].groupby('year')
	y = group.apply(lambda x: x['Vowels'].mean())
	x = group.apply(lambda x: x['year'].mean())
	ax.plot(x,y,color='k',linestyle='dashed')
	if ax is None: plt.show()

def plot_vowel_fraction_usa(ax=None):
	df = pd.read_csv('name_statistics_USA.csv')
	df['Vowels'] = df['name'].apply(lambda x: vowel_count(x)/len(x))

	if ax is None: fig,ax = plt.subplots(1)
	group = df.loc[df['gender']=='F'].groupby('year')
	y = group.apply(lambda x: x['Vowels'].mean())
	x = group.apply(lambda x: x['year'].mean())
	ax.plot(x,y,color='green')

	group = df.loc[df['gender']=='M'].groupby('year')
	y = group.apply(lambda x: x['Vowels'].mean())
	x = group.apply(lambda x: x['year'].mean())
	ax.plot(x,y,color='green',linestyle='dashed')
	if ax is None: plt.show()

def plot_vowel_fraction_dutch(ax=None):
	df = pd.read_csv('name_statistics_netherlands_1880-2014.csv')
	df['Vowels'] = df['name'].apply(lambda x: vowel_count(x)/len(x))

	if ax is None: fig,ax = plt.subplots(1)
	group = df.loc[df['gender']=='vrouw'].groupby('year')
	y = group.apply(lambda x: x['Vowels'].mean())
	x = group.apply(lambda x: x['year'].mean())
	ax.plot(x,y,color='orange')

	group = df.loc[df['gender']=='man'].groupby('year')
	y = group.apply(lambda x: x['Vowels'].mean())
	x = group.apply(lambda x: x['year'].mean())
	ax.plot(x,y,color='orange',linestyle='dashed')
	plot_vowel_fraction(ax)
	plot_vowel_fraction_UK(ax)
	plot_vowel_fraction_belgium(ax)
	plot_vowel_fraction_usa(ax)
	ax.set_xlim([1920,2020])
	ax.set_ylabel('Average fraction of vowels',fontsize=16)
	ax.set_xlabel("Year",fontsize=16)
	if ax is None: plt.show()

#plot_vowel_fraction()
fig,ax = plt.subplots(1,2,figsize=(11,6))
plt.subplots_adjust(wspace=0.25)
plot_average_length(ax[0])
plot_vowel_fraction_dutch(ax[1])
plt.savefig('name_length_vowels.png')
plt.show()
#read_italy()


