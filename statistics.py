import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import IPython
from scipy.interpolate import interp1d
import glob
from scipy.integrate import trapz
from matplotlib.ticker import MaxNLocator
import re
import unidecode

def weighted_median(data, weights,return_quantiles=False):
    """
    Args:
      data (list or numpy.array): data
      weights (list or numpy.array): weights
    """
    data, weights = np.array(data).squeeze(), np.array(weights).squeeze()
    s_data, s_weights = map(np.array, zip(*sorted(zip(data, weights))))
    midpoint = 0.5 * sum(s_weights)
    q1, q3 = 0.25*sum(s_weights), 0.75*sum(s_weights)
    if any(weights > midpoint):
        w_median = (data[weights == np.max(weights)])[0]
    else:
        cs_weights = np.cumsum(s_weights)
        idx = np.where(cs_weights <= midpoint)[0][-1]
        idx1 = np.where(cs_weights <= q1)[0][-1]
        idx3 = np.where(cs_weights <= q3)[0][-1]
        if cs_weights[idx] == midpoint:
            w_median = np.mean(s_data[idx:idx+2])
        else:
            w_median = s_data[idx+1]
    if return_quantiles:
    	return w_median,s_data[idx1],s_data[idx3]
    return w_median

def get_survivors(x,gender='male',current_year=2019):
	if gender == 'male':
		lab = 'Mannen'
	else:
		lab = 'Vrouwen'
	if x >= 1900 and x <=1910:
		tafel = pd.read_csv('Overlevingstafels/1900-1910.csv',sep=';')
	elif x >= 1911 and x <= 1930:
		tafel = pd.read_csv('Overlevingstafels/1911-1930.csv',sep=';')
	elif x >= 1931 and x <= 1949:
		tafel = pd.read_csv('Overlevingstafels/1931-1949.csv',sep=';')
	elif x >= 1950 and x <= 1974:
		tafel = pd.read_csv("Overlevingstafels/1950-1974.csv",sep=';')
	elif x >= 1975 and x <= 1999:
		tafel = pd.read_csv("Overlevingstafels/1975-1999.csv",sep=';')
	elif x >= 2000 and x <= 2020:
		tafel = pd.read_csv("Overlevingstafels/2000-2020.csv",sep=';')
	else:
		return 0

	years = np.array([])
	for i in tafel['Geslacht']:
		years = np.append(years,int(i))

	ind = np.where(years==x)[0][0]

	if current_year - x <= 0:
		pass
	elif current_year -x >= 99:
		lab +='.99'
	else:
		lab += '.%d'%(current_year-x)

	return float(tafel[lab][ind]) / 100000

def median_ages():
	files = glob.glob('histograms/*csv')
	data_out = {'name':[],'gender':[],'median age':[],'1st quartile':[],'3rd quartile':[],'fraction died':[]}
	for fi in files:
		data = pd.read_csv(fi)
		f = interp1d(data.years,data.frequency,bounds_error=False,fill_value='extrapolate')
		surv = np.array([])
		yrs2 = np.arange(np.floor(data.years.min()),np.ceil(data.years.max()))
		for yr in yrs2:
			surv = np.append(surv,get_survivors(yr))
		f = interp1d(yrs2,surv,bounds_error=False,fill_value='extrapolate')

		med_old = weighted_median(data.years,data.frequency)
		med_new,q1,q3 = weighted_median(data.years,data.frequency*f(data.years),return_quantiles=True)

		use = data.years>=1900
		fraction_died = 1-trapz(data.frequency[use]*f(data.years[use]),data.years[use])/trapz(data.frequency[use],data.years[use])
	
		name = re.search('histograms/(.*)_(.*).csv',fi).group(1)
		gender = re.search('histograms/(.*)_(.*).csv',fi).group(2)

		data_out['name'].append(name)
		data_out['gender'].append(gender)
		data_out['median age'].append(med_new)
		data_out['1st quartile'].append(q1)
		data_out['3rd quartile'].append(q3)
		data_out['fraction died'].append(fraction_died)
	pd.DataFrame(data_out).to_csv('median_ages_netherlands.csv',index=False)

def median_age(name,gender,years,frequency):
	f = interp1d(years,frequency,bounds_error=False,fill_value='extrapolate')
	surv = np.array([])
	yrs2 = np.arange(np.floor(years.min()),np.ceil(years.max()))
	for yr in yrs2:
		surv = np.append(surv,get_survivors(yr))
	f = interp1d(yrs2,surv,bounds_error=False,fill_value='extrapolate')

	med_old = weighted_median(years,frequency)
	med_new,q1,q3 = weighted_median(years,frequency*f(years),return_quantiles=True)

	use = data.years>=1900
	fraction_died = 1-trapz(frequency[use]*f(years[use]),years[use])/trapz(frequency[use],years[use])
	
	return med_new,q1,q3,fraction_died

def merge():
	"""
	Merge median ages and other properties to create a master catalog.
	"""
	popularity = pd.read_csv('name_popularity_netherlands_1880-2014.csv')
	med_age = pd.read_csv('median_ages_netherlands.csv')
	mask = popularity.gender == 'M'
	popularity.loc[mask,'gender'] = 'man'
	popularity.loc[~mask,'gender'] = 'vrouw'
	popularity.name = [unidecode.unidecode(name) for name in popularity.name]
	master = pd.merge(med_age,popularity,on=['name','gender'])
	master.to_csv('name_statistics_netherlands_1880-2014.csv')

def _gini(x):
    # (Warning: This is a concise implementation, but it is O(n**2)
    # in time and memory, where n = len(x).  *Don't* pass in huge
    # samples!)

    # Mean absolute difference
    mad = np.abs(np.subtract.outer(x, x)).mean()
    # Relative mean absolute difference
    rmad = mad/np.mean(x)
    # Gini coefficient
    g = 0.5 * rmad
    return g

def _hoover(x):
    meanx = np.mean(x)
    return 0.5*np.sum(abs(x-meanx))/np.sum(x)

def _galt(x):
	return x.max()/np.median(x)

def _palma(x):
	x = np.sort(x)
	return np.sum(x[-10:])/np.sum(x[0:40])

def _ratio2020(x):
	x = np.sort(x)
	return np.sum(x[-20:])/np.sum(x[0:20])

def inequality_metrics():
	"""
	Calculate inequality metrics & merge with main catalog.
	Note: these are computed for the 100 most popular names in a given year. 
	---
	Palma ratio ('richest' 10% over 'poorest' 40%)
	2020 ratio (20% to 20%)
	Hoover index (how much needs to be redistributed to be perfectly equal?)
	Galt index: ratio of the richest ('CEO') to median worker
	Gini index: most common inequality metric, area under Lorentzian
	"""
	df = pd.read_csv('name_statistics_netherlands_1880-2014.csv')
	df['Gini'] = np.nan 
	df['Galt'] = np.nan 
	df['Hoover'] = np.nan 
	df['Palma'] = np.nan 
	df['ratio2020'] = np.nan

	df['Gini male'] = np.nan 
	df['Galt male'] = np.nan 
	df['Hoover male'] = np.nan 
	df['Palma male'] = np.nan 
	df['ratio2020 male'] = np.nan

	df['Gini female'] = np.nan 
	df['Galt female'] = np.nan 
	df['Hoover female'] = np.nan 
	df['Palma female'] = np.nan 
	df['ratio2020 female'] = np.nan

	for yr in df.year.unique():
		df['Gini'].loc[df.year==yr] = _gini(df['aantal'].loc[df.year==yr])
		df['Hoover'].loc[df.year==yr] = _hoover(df['aantal'].loc[df.year==yr])
		df['Galt'].loc[df.year==yr] = _galt(df['aantal'].loc[df.year==yr])
		df['Palma'].loc[df.year==yr] = _palma(df['aantal'].loc[df.year==yr])
		df['ratio2020'].loc[df.year==yr] = _ratio2020(df['aantal'].loc[df.year==yr])

		df['Gini male'].loc[(df.year==yr)&(df.gender=='man')] = _gini(df['aantal'].loc[(df.year==yr)&(df.gender=='man')])
		df['Hoover male'].loc[(df.year==yr)&(df.gender=='man')] = _hoover(df['aantal'].loc[(df.year==yr)&(df.gender=='man')])
		df['Galt male'].loc[(df.year==yr)&(df.gender=='man')] = _galt(df['aantal'].loc[(df.year==yr)&(df.gender=='man')])
		df['Palma male'].loc[(df.year==yr)&(df.gender=='man')] = _palma(df['aantal'].loc[(df.year==yr)&(df.gender=='man')])
		df['ratio2020 male'].loc[(df.year==yr)&(df.gender=='man')] = _ratio2020(df['aantal'].loc[(df.year==yr)&(df.gender=='man')])

		use = (df.year == yr)&(df.gender=='vrouw')
		df['Gini female'].loc[use] = _gini(df['aantal'].loc[use])
		df['Hoover female'].loc[use] = _hoover(df['aantal'].loc[use])
		df['Galt female'].loc[use] = _galt(df['aantal'].loc[use])
		df['Palma female'].loc[use] = _palma(df['aantal'].loc[use])
		df['ratio2020 female'].loc[use] = _ratio2020(df['aantal'].loc[use])

	#Update the master file
	df.to_csv('name_statistics_netherlands_1880-2014.csv')

def name_length():
	cat = pd.read_csv('name_statistics_netherlands_1880-2014.csv',index_col=0)
	cat['Name length, M'] = np.nan
	cat['Name length, F'] = np.nan
	cat['Name length, M'].loc[cat.gender=='man'] = [len(name) for name in cat['name'].loc[cat.gender=='man']]
	cat['Name length, F'].loc[cat.gender=='vrouw'] = [len(name) for name in cat['name'].loc[cat.gender=='vrouw']]
	cat['Name length'] = [len(name) for name in cat['name']]
	cat['Weighted name length'] = np.nan
	cat['Weighted name length, M'] = np.nan
	cat['Weighted name length, F'] = np.nan

	for yr in cat['year'].unique():
		cat['Weighted name length'].loc[cat.year==yr] = np.average(cat['Name length'].loc[cat.year==yr],weights=cat['aantal'].loc[cat.year==yr])
		use = (cat.year==yr)&(cat.gender=='man')
		cat['Weighted name length, M'].loc[use] = np.average(cat['Name length'].loc[use],weights=cat['aantal'].loc[use])
		use = (cat.year==yr)&(cat.gender=='vrouw')
		cat['Weighted name length, F'].loc[use] = np.average(cat['Name length'].loc[use],weights=cat['aantal'].loc[use])
	cat.to_csv('name_statistics_netherlands_1880-2014.csv')

if __name__ == "__main__":
	#median_ages()
	#merge()
	#inequality_metrics()
	name_length()
