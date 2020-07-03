import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib
import pandas as pd
import IPython
from scipy.interpolate import interp1d
import glob
from scipy.integrate import trapz
from matplotlib.ticker import MaxNLocator
from read_histograms import download_image
import re
from statistics import get_survivors, weighted_median
import unidecode
import seaborn as sns

def US_comparison():
	pass

def show_name_lengths():
	cat = pd.read_csv('name_statistics_netherlands_1880-2014.csv')

	fig,ax = plt.subplots(1)
	means_male = cat.loc[cat['Name length, M'].notna()].groupby('year').apply(lambda x: x['Name length, M'].mean())
	ax.plot(means_male.index,means_male,color='blue',label='Gemiddelde, man')
	weighted_mean_male = cat.loc[cat['Weighted name length, M'].notna()].groupby('year').apply(lambda x: x['Weighted name length, M'].mean())
	ax.plot(means_male.index,weighted_mean_male,linestyle='dashed',color='blue',label='Gewogen gemiddelde, man')
	means_female = cat.loc[cat['Weighted name length, F'].notna()].groupby('year').apply(lambda x: x['Name length, F'].mean())
	ax.plot(means_female.index,means_female,color='violet',label='Gemiddelde, vrouw')
	weighted_mean_female = cat.loc[cat['Weighted name length, F'].notna()].groupby('year').apply(lambda x: x['Weighted name length, F'].mean())
	ax.plot(means_female.index,weighted_mean_female,linestyle='dashed',color='violet',label='Gewogen gemiddelde, vrouw')
	ax.set_xlabel('Jaren',fontsize=14)
	ax.set_ylabel('Gemiddelde naamlengte',fontsize=14)
	ax.legend(fontsize=14)
	ax.set_frame_on(False)
	ax.grid()
	plt.savefig('images/average_name_length.png')
	plt.show()


def show_popularity_distributions():
	cat = pd.read_csv('name_statistics_netherlands_1880-2014.csv').sort_values('rank')
	yrs = cat['year'].unique()
	c = np.arange(1,yrs.size+1)
	norm = matplotlib.colors.Normalize(vmin=c.min(), vmax=c.max())
	cmap = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.Blues)
	cmap.set_array([])

	norm = matplotlib.colors.Normalize(vmin=c.min(), vmax=c.max())
	cmap2 = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.Reds)
	cmap2.set_array([])

	fig,ax = plt.subplots(1,2,figsize=(12,5))
	for i,yr in enumerate(yrs):
		tmp = cat.loc[(cat.year==yr)&(cat.gender=='man')]
		ax[0].plot(tmp['rank'],tmp.aantal,alpha=0.3,c=cmap.to_rgba(c.max()-i))

		tmp = cat.loc[(cat.year==yr)&(cat.gender=='vrouw')]
		ax[1].plot(tmp['rank'],tmp.aantal,alpha=0.3,c=cmap2.to_rgba(c.max()-i))
	ax[0].set_yscale('log')
	cbar = plt.colorbar(cmap,ax=ax[0],ticks=c[::20])#+yrs.min()-1)
	cbar.ax.set_yticklabels([str(int(tick.get_text())+int(yrs.min()-1)) for tick in cbar.ax.get_yticklabels()])
	cbar.set_clim([c.min(),c.max()])
	ax[0].set_xlabel('Rank',fontsize=20)
	ax[0].set_ylabel('Frequency',fontsize=20)
	ax[0].set_facecolor('grey')
	#ax[0].set_frame_on(False)
	ax[0].grid(which='both')

	ax[1].set_yscale('log')
	cbar = plt.colorbar(cmap2,ax=ax[1],ticks=c[::20])#+yrs.min()-1)
	cbar.set_clim([c.min(),c.max()])
	cbar.ax.set_yticklabels([str(int(tick.get_text())+int(yrs.min()-1)) for tick in cbar.ax.get_yticklabels()])
	ax[1].set_xlabel('Rank',fontsize=20)
	ax[1].set_ylabel('Frequency',fontsize=20)
	ax[1].set_facecolor('grey')
	ax[1].grid(which='both')
	plt.savefig('images/Frequency_over_time.png')

	c = np.arange(1,yrs.size+1)
	norm = matplotlib.colors.Normalize(vmin=c.min(), vmax=c.max())
	cmap = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.Blues)
	cmap.set_array([])

	norm = matplotlib.colors.Normalize(vmin=c.min(), vmax=c.max())
	cmap2 = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.Reds)
	cmap2.set_array([])

	#What we have forgotten above: we want to normalize by the number of babies born in a year...!
	fig,ax = plt.subplots(1,2,figsize=(12,5))
	for i,yr in enumerate(yrs):
		tmp = cat.loc[(cat.year==yr)&(cat.gender=='man')]
		ax[0].plot(tmp['rank'],tmp.aantal/tmp.aantal.max(),alpha=0.8,c=cmap.to_rgba(c.max()-i))

		tmp = cat.loc[(cat.year==yr)&(cat.gender=='vrouw')]
		ax[1].plot(tmp['rank'],tmp.aantal/tmp.aantal.max(),alpha=0.8,c=cmap2.to_rgba(c.max()-i))
	ax[0].set_yscale('log')
	cbar = plt.colorbar(cmap,ax=ax[0],ticks=c[::20])#+yrs.min()-1)
	cbar.ax.set_yticklabels([str(int(tick.get_text())+int(yrs.min()-1)) for tick in cbar.ax.get_yticklabels()])
	cbar.set_clim([c.min(),c.max()])
	ax[0].set_xlabel('Rank',fontsize=20)
	ax[0].set_ylabel('Relative frequency',fontsize=20)
	ax[0].set_facecolor('grey')
	#ax[0].set_frame_on(False)
	ax[0].grid(which='both')

	ax[1].set_yscale('log')
	cbar = plt.colorbar(cmap2,ax=ax[1],ticks=c[::20])#+yrs.min()-1)
	cbar.ax.set_yticklabels([str(int(tick.get_text())+int(yrs.min()-1)) for tick in cbar.ax.get_yticklabels()])
	ax[1].set_xlabel('Rank',fontsize=20)
	ax[1].set_ylabel('Relative frequency',fontsize=20)
	ax[1].set_facecolor('grey')
	#ax[0].set_frame_on(False)
	ax[1].grid(which='both')
	plt.savefig('images/Relative_frequency_over_time.png')
	plt.show()

def show_babyboom():
	cat = pd.read_csv('name_statistics_netherlands_1880-2014.csv').groupby('year').agg('sum')

	fig,ax = plt.subplots(1)
	ax.plot(cat.index,cat['aantal']/1e3)
	ax.set_ylabel('Annual number of births, /1000',fontsize=16)
	ax.set_xlabel('Year',fontsize=16)
	ax.grid()
	ax.text(1946,cat['aantal'][cat.index==1946]/1e3,'Babyboom',fontsize=14)
	ax.set_frame_on(False)
	plt.savefig('images/annual_births.png')
	plt.show()

def inequality():
	"""Use inequality metrics to analyse naming traditions."""
	cat = pd.read_csv('name_statistics_netherlands_1880-2014.csv')
	men = cat.loc[cat['Gini male'].notna()].groupby('year')
	women =	cat.loc[cat['Gini female'].notna()].groupby('year')
	both = cat.loc[cat['Gini'].notna()].groupby('year')

	indices = ['Gini','Galt','Hoover','Palma','ratio2020']
	for index in indices:
		fig,ax = plt.subplots(1)
		gini = men.apply(lambda x: x[index+' male'].median())
		ax.plot(gini.index,gini.values,color='blue',label='Male')
		gini = women.apply(lambda x: x[index+' female'].median())
		ax.plot(gini.index,gini.values,color='violet',label='Female')
		gini = both.apply(lambda x: x[index].median())
		ax.plot(gini.index,gini.values,color='k',label='All')
		ax.set_xlabel('Year',fontsize=16)
		ax.set_ylabel(index,fontsize=16)
		ax = plt.gca()
		plt.grid(which='both')
		ax.set_frame_on(False)
		plt.legend()
		plt.savefig('images/%s.png'%index)

def violinplot():
	names = ['Alexander','Jayden','Noah','Jack','Daan','Mohamed','Kevin'\
				,'David','Simon','Marco','Ronald','Dirk','Gerrit','Johannes','Theodorus']
	yrs = np.arange(1880,2015)
	surv = interp1d(yrs,[get_survivors(yr,gender='male') for yr in yrs],bounds_error=False,fill_value=0)
	df = pd.DataFrame({'years':[]})
	for name in names:
		tmp = pd.read_csv('histograms/%s_man.csv'%name)
		interp_grid = interp1d(tmp['years'],tmp['frequency'],bounds_error=False,fill_value=0)
		df[name] = interp_grid(yrs)*surv(yrs)
	IPython.embed()

	sns.violinplot(df)

def boxplot(gender='man'):
	"""
	Plot the ages of the people with a selection of names.
	"""
	cat = pd.read_csv('name_statistics_netherlands_1880-2014.csv')
	cat = cat.sort_values('median age')
	if gender == 'man':
		names = ['Alexander','Jayden','Noah','Jack','Daan','Mohamed','Kevin'\
				,'David','Simon','Marco','Ronald','Dirk','Gerrit','Johannes','Theodorus']
		clr = 'lightblue'
	else: 
		gender = 'vrouw'
		names = ['Noa','Maud','Julia','Emma','Femke','Sanne',\
				'Laura','Kelly','Chantal','Esther','Jacqueline','Jolanda',\
				'Sophia','Wilma','Johanna','Maria','Theodora','Trijntje']
		clr = 'violet'
	group = cat.loc[(cat.gender==gender)&(cat.name.isin(names))].groupby('name',sort=False)

	x = group.apply(lambda x: x['median age'].unique()[0])
	y = np.arange(x.size)
	low, high =group.apply(lambda x: x['1st quartile'].unique()[0]),group.apply(lambda x: x['3rd quartile'].unique()[0])
	labels = x.index

	fig,ax = plt.subplots(1,figsize=(7,10))
	plt.subplots_adjust(top=0.91,bottom=0.06,right=0.95)
	ax.hlines(np.arange(x.size),xmin=1900,xmax=2020,color='gray',linestyle='dotted',alpha=0.7,linewidth=1)
	ax.scatter(x=x,y=y,facecolors='firebrick',edgecolor='k',s=50,zorder=5,lw=2)
	x2 = np.array([low,high])
	y2 = np.vstack((y,y))
	ax.plot(x2,y2,color='k',lw=21)
	ax.plot(x2,y2,color=clr,lw=20)

	ax.text(x2[0,5],y2[0,5]-0.2,'25e',fontsize=14,fontweight='bold',zorder=20)
	ax.text(x2[1,5]-6.5,y2[1,5]-0.2,'75e',fontsize=14,fontweight='bold',zorder=20)

	ax.set_yticks(np.arange(x.size))
	ax.set_yticklabels(labels)
	ax.set_xlim(1935,2020)
	ax.set_frame_on(False)
	ax2 = ax.twiny()	
	ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
	ax2.set_xlim(1935,2020)
	ax2.set_frame_on(False)
	labs = ax2.get_xticks()
	ax2.set_xticklabels(2020-labs)
	ax2.grid(linestyle='dotted',zorder=15)
	ax.text(1933,x.size+0.8,'Huidige leeftijd (2020)',fontweight='bold')
	ax.set_xlabel('Geboortejaar',fontsize=16)
	plt.savefig('age_distribution_%s.png'%gender)
	plt.show()

def plot_age_distr(name,gender,ax=None):
	"""
	Plot the distribution of births, plus
	the survivors (expected number that are still alive, 
	given average survival numbers)
	"""
	data = pd.read_csv('histograms/%s_%s.csv'%(name,gender))
	f = interp1d(data['years'],data['frequency'],bounds_error=False,fill_value='extrapolate')
	surv = np.array([])
	yrs2 = np.arange(np.floor(data['years'].min()),np.ceil(data['years'].max()))
	for yr in yrs2:
		surv = np.append(surv,get_survivors(yr))
	f = interp1d(yrs2,surv,bounds_error=False,fill_value='extrapolate')

	if gender == 'female' or gender == 'vrouw':
		clr = 'violet'
		clr2 = 'darkviolet'
	else:
		clr = 'lightblue'
		clr2 = 'darkblue'

	med_old = weighted_median(data.years,data.frequency)
	med_new,q1,q3 = weighted_median(data.years,data.frequency*f(data.years),return_quantiles=True)
	if ax is None:
		fig,ax = plt.subplots(1)
	ax.set_facecolor((0.5, 0.95, 0.95))

	ax.plot(data.years,data.frequency/data.frequency.max(),color='k')
	ax.plot(data.years,data.frequency*f(data.years)/data.frequency.max(),color=clr2)
	ax.bar(data.years,data.frequency*f(data.years)/data.frequency.max(),color=clr,width=1)

	#ax.bar(yrs,height*f(yrs),color='blue')
	ax.set_xlabel('Year',fontsize=16)#Year',fontsize=16)
	ax.set_ylabel('Relative frequency',fontsize=16)#Relative frequency',fontsize=16)
	#ax.set_title('%s, median age: %.1f years'%(name,2018-med_new),fontsize=20)
	ax.text(1865,1.1,'Age distribution of Dutch people with the name',fontsize=12)
	ax.text(1971,1.1,'%s'%name,fontweight='heavy',fontsize=16)
	ax.set_ylim(ax.get_ylim())
	ymax = interp1d(data.years,data.frequency*f(data.years)/data.frequency.max(),bounds_error=False,fill_value=0)(med_new)
	ax.plot([med_new,med_new],[0,ymax],color=clr2,label='Median age',lw=3)
	#ax.plot([q1,q1],ax.get_ylim(),color='k',linestyle='dashed')
	#ax.plot([q3,q3],ax.get_ylim(),color='k',linestyle='dashed')
	ax.set_xlim(1880,2020)#ax.get_xlim())
	#ax.grid(zorder=0)
	if name == 'Johannes' or name == 'Maria':
		ymax1 = interp1d(data.years,data.frequency,bounds_error=False,fill_value=0)(1903)/data.frequency.max()-0.01
		ax.annotate(s='', xy=(1903,ymax1), xytext=(1903,0), arrowprops=dict(arrowstyle='<->',lw=2))
		ax.text(1906,0.1,'Still alive',fontsize=12,color=clr2)
		ax.text(1881,ymax1+0.05,'All births',fontsize=12,color='k')
		ax.annotate(s='Median Age', xy=(med_new,ymax1), xytext=(1980,0.7), arrowprops=dict(arrowstyle='->',lw=2),color=clr2)

	if name == 'Jayden':
		ax.text(1934,0.5,"Vrijwel alle Jayden's \nzijn na 2000 geboren.\nDe gemiddelde Jayden\nis nu %d jaar oud"%(2020-med_new),fontsize=16,color=clr2)

	if name == 'Maud':
		ax.text(1904,0.5,"Maud nam al sinds halverwege\nde jaren 80 toe in populariteit,\nmaar piekte na 2000.\n\nDe gemiddelde Maud\nis nu %d jaar oud"%(2020-med_new),fontsize=16,color=clr2)

	#ax2 = ax.twiny()	
	#ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
	#ax2.set_xlim(ax.get_xlim())
	#ax2.set_frame_on(False)
	#labs = ax2.get_xticks()
	#ax2.set_xticklabels(2020-labs)
	ax.grid(zorder=0)
	ax.set_frame_on(False)
	plt.savefig('Johannes_Johanna.png',dpi=100)
	#if ax is None:
	#	plt.savefig('%s.png'%name,dpi=80)

def plot_age_distr_male_female():
	fig,ax = plt.subplots(1,2,figsize=(15,7))
	plot_age_distr('Johannes','man',ax[0])
	plot_age_distr('Maria','vrouw',ax[1])
	plt.show()

if __name__ == "__main__":
	#plot_age_distr_male_female()
	#boxplot()
	#boxplot('vrouw')
	#inequality()
	#show_popularity_distributions()
	#show_name_lengths()
	violinplot()

