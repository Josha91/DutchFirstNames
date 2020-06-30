"""
This script can be used to (1) download histograms of the frequency of a name
(given the URL) over time. 
This is necessary because the data is not provided in anything but the
graphic form. 

Data from Meertens institute Voornamenbank
"""

import glob
import IPython
import itertools
import matplotlib.pyplot as plt 
import numpy as np 
import os
import pandas as pd
from PIL import Image
from read_popularity_lists import download_all
import urllib.request
import unidecode

def detect_axes(image):
	"""Detect the location of the axes.
	"""
	sh = image.shape

	#First: horizontal axes. 
	#The line with the most zero values is thought to be the axis.  
	zeros = []
	for line in range(sh[0]):
		#if more than 75% of this line is not 0, it's not the axis.
		#if np.sum(image[line,:]==0) < 0.75*sh[0]: continue 
		zeros.append(np.sum(image[line,:]==0))
	zeros = np.asarray(zeros)
	x_axis = np.where(zeros==zeros.max())[0][0]

	#The y-axis is the first line where > 75 % is 0.
	zeros = []
	for line in range(sh[1]):
		zeros_fraction = np.sum(image[:,line]==0)/image.shape[0]
		if zeros_fraction > 0.75: break
	#zeros = np.asarray(zeros)
	y_axis = line#np.where(zeros==zeros.max())[0][0]
	return x_axis,y_axis

def get_positions_old(image,xaxis):
	#based on a test image, all the bars should be visible 
	#3 pixels above the xaxis. 
	#In the automatic download, there is actually no space between the bars.
	#this won't work. 
	arr = image[xaxis-3,:]
	groups,uniquekeys = [],[]
	tot_len =0
	for k,g in itertools.groupby(arr,key=lambda i: i==0):
		groups.append(list(g))
		uniquekeys.append(k)

	index = []
	tot = 0
	for k,g in itertools.groupby(arr,key=lambda i:i==0):
		index.append(tot)
		tot +=len(list(g))

	locs = []
	for i,group in enumerate(groups):
		if len(group) >=6 and np.sum(group) == 0:
			locs.append(index[i]+len(group)/2)

	return np.asarray(locs)

def detect_ticks(image,xaxis,yaxis):
	"""Detect the positions of the ticks"""
	ticklabels = np.arange(1890,2030,10)

	arr = image[xaxis+2,:] #A row in the image where you would find 'dips' for ticks
	groups,uniquekeys = [],[]
	tot_len = 0
	for k,g in itertools.groupby(arr,key=lambda i: i==0):
		groups.append(list(g))
		uniquekeys.append(k)

	index = []
	tot = 0
	for k,g in itertools.groupby(arr,key=lambda i:i==0):
		index.append(tot)
		tot +=len(list(g))

	locs = []
	for i,group in enumerate(groups):
		if len(group) == 1 and np.sum(group) == 0:
			locs.append(index[i])
	locs = np.asarray(locs)
	use = locs > yaxis+10
	if np.sum(use) > 14: IPython.embed()
	if np.sum(use) > 14: raise ValueError #Too many ticks, something went wrong
	return ticklabels,locs[use]

def return_histogram(image,locs,ticklabels,ticklocs):
	"""
	We found the locations, now just read off the values and return
	"""
	dx = ticklocs[1]-ticklocs[0]
	dt = 10/dx #years per pixel.

	years = (locs-ticklocs[0])*dt  + ticklabels[0]

	height = []
	for xloc in locs:
		arr = image[:,int(xloc)]
		groups = []
		for k,g in itertools.groupby(arr,key=lambda i: i==0):
			groups.append(list(g))
		max_length = 0
		for g in groups:
			if np.sum(g) == 0:
				if len(g) > max_length:
					max_length = len(g)
		height.append(max_length)
	return years,np.asarray(height)

def get_histogram(img_path):
	"""
	Read in a histogram from an image and save as .csv
	"""
	im = Image.open(img_path)
	size = im.size

	#split in different bands
	source = im.split() #R, G, B

	#select regions where red is less than 100
	mask = source[0].point(lambda i: i< 100 and 255)

	#process the green band
	out = source[1].point(lambda i: i*0.7)

	#paste the processed band back, but only where red was < 100
	source[1].paste(out,None,mask)
	im = Image.merge(im.mode,source)

	im = np.asarray(im)
	im_or = Image.fromarray(im)

	#to detect a bar:
	xax,yax = detect_axes(im[:,:,2])
	#Right until here. Positions aren't recognised...
#	positions = get_positions(im[:,:,2],xax)
	ticklabels,ticklocs = detect_ticks(im[:,:,2],xax,yax)
	positions = np.arange(yax+2,im[:,:,2].shape[1]-20,5)
	years,height = return_histogram(im[:,:,2],positions,ticklabels,ticklocs)
	#detect the *width* by checking the interval at which the image dips in the horizontal derection.
	#Detect the relative height by checking the continuous vertical length of each of those positions. 
	#detect the year of each by detecting the position of the y-axis.  
	#then detect the pixel->time conversion. 
	df = pd.DataFrame({'years':years,'frequency':height})
	return df

def download_images():
	if not os.path.exists('name_popularity_netherlands_1880-2014.csv'):
		download_all() #download the popularity lists. 

	popcsv = pd.read_csv('name_popularity_netherlands_1880-2014.csv')
	#unique names that occured in the top 100 at some point in the period 1880-2014
	popcsv.drop_duplicates('naam') 

	data = pd.DataFrame({'years':[],'frequency':[]})
	for i in range(popcsv.shape[0]):
		name = popcsv['naam'][i]
		unaccented = unidecode.unidecode(name)
		if popcsv['tag'][i] == 'M':
			gender = 'man'
		else:
			gender = 'vrouw'
		if os.path.exists('histograms/%s_%s.jpg'%(name,gender)): continue
		try:
			link = 'https://www.meertens.knaw.nl/nvb/populariteit/absoluut/%s/afbeelding/naam/%s'%(gender,unaccented)
			urllib.request.urlretrieve(link, "histograms/%s_%s.jpg"%(name,gender))
			df = get_histogram('histograms/%s_%s.jpg'%(name,gender))
			data.to_csv('histograms/%s_%s.csv'%(name,gender))
		except: #problems occur with accents.
			print('An error occurred when trying to read:')
			print(name,gender)

if __name__ == '__main__':
	download_images()
