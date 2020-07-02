# DutchFirstNames
Analysis of babynames in the Netherlands over the period 1880-2014.

Data is taken from de Voornamenbank from the Meertens Institute in the Netherlands. They publish lists of the  most popular names in a given year (as far back as 1880), as well as the popularity of a given name through time. 

The former is read in with simple scraping of the table. The latter is only published in a histogram, so the read_histograms.py script in this repository reads in the data by downloading the image and analysing it. This is achieved in a few steps:
(1) Identify the row (column) of the x-axis (y-axis), as rows of zeros in the R-channel
(2) Identify ticks on the x-axis as period zeropoints in the row below the x-axis. Times are extracted relative to the ticks.
(3) At every time, identify the height (in pixels) of the histogram by continuous segments of zeros, from the x-axis.
(4) Currently, the histograms are relative: no y-axis tick extraction is performed. 

After reading in the data, analysis is performed (TBD)

Required Python packages:
glob, itertools, matplotlib, numpy, os, pandas, PIL, urllib, unidecode

``` r
URL = "https://www.ssa.gov/oact/babynames/names.zip"
dir.create("data")
download.file(URL, destfile = "./data/babyname.zip")
unzip("./data/babyname.zip", exdir = "./data")
```

Code
