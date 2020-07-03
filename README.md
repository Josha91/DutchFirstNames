# DutchFirstNames
Analysis of babynames in the Netherlands over the period 1880-2014.

Data is taken from de Voornamenbank from the Meertens Institute in the Netherlands. They publish lists of the  most popular names in a given year (as far back as 1880), as well as the popularity of a given name through time. 

The former is read in with simple scraping of the table. The latter is only published in a histogram, so the read_histograms.py script in this repository reads in the data by downloading the image and analysing it. This is achieved in a few steps:
1. Automated downloading of image
1. Identify the row (column) of the x-axis (y-axis), as rows of zeros in the R-channel
1. Identify ticks on the x-axis as period zeropoints in the row below the x-axis. Times are extracted relative to the ticks.
1. At every time, identify the height (in pixels) of the histogram by continuous segments of zeros, from the x-axis.
1. Currently, the histograms are relative: no y-axis tick extraction is performed. 
These tasks are performed in `read_popularity_lists.py` and `read_histograms.py`

After reading this in, some statistics are calculated and appended to the dataframe: 
1. Characterlength of a given name
1. The average characterlength of babynames in a given year; the average is weighted by a name's popularity
1. Inequality statistics of babynames in a given year. The calculated metrics are the Gini index, the Hoover index, the Palma ratio, the 20-20 ratio, and the Galt score. These are useful because it tells us something about the dominance of the most popular name(s) - see analysis below. 
1. The birth distribution of people with a given name is combined with average survival statistics from the central Dutch bureau of statistics (CBS), to obtain the average age of the people with that name that are still living. See analysis below. 
These tasks are performed by `statistics.py`

The main output data can be found in `name_statistics_netherlands_1880-2014.csv`

Required Python packages:
glob, itertools, matplotlib, numpy, os, pandas, PIL, seaborn, urllib, unidecode

Let's have a look at the dataset. 

The zeroth-order thing we can do, is look at the total number of births over time. A small caveat here is that this only tracks the top-100 names in boys and girls (~200 names in total). The percentage of the total births that these reprents is unlikely to be constant over time (in particular, I would expect this to be a smaller percentage in recent years, as we will see later). Adding the total number of births would be a small, straightforward extension of the dataset. 
Most notably, in this figure we see:
1. Rising total births early on, corresponding to a rising population size. 
1. The babyboom after the war is a significant spike in the number of births. 
1. The number of births falls sharply after the 1960s, coinciding with the introduction of advanced anti-conception.
1. At the beginning of the 21st century the birth rates are similar to those at the beginning of the 20th century, but with a much larger population this represents a much lower fertility. 

![Annual births of top-100 male and female names](https://github.com/Josha91/DutchFirstNames/blob/master/images/annual_births.png)



