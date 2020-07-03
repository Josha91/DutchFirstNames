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

Required Python packages:
glob, itertools, matplotlib, numpy, os, pandas, PIL, urllib, unidecode

``` r
URL = "https://www.ssa.gov/oact/babynames/names.zip"
dir.create("data")
download.file(URL, destfile = "./data/babyname.zip")
unzip("./data/babyname.zip", exdir = "./data")
```

Code

![This is about asterix](https://github.com/Josha91/GoodScraping/blob/master/asterix_scores.png)

![](../GoodScraping/asterix_scores.png)
