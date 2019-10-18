#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 13:23:18 2019

@author: Nathaniel
"""
#General lib Import
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)
import os as os
import matplotlib as mp
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import seaborn as sb
import collections
from collections import Counter
import re

#######################################DATA###################################
#import data files and describe data
movies = pd.read_csv('/Volumes/NO NAME/TMC_WalmartTest/data/movies.csv', header = 0)
print(movies.info())
movies.head()
#9742 entries
#3 fileds - movieId    9742 non-null int64 |title      9742 non-null object | genres     9742 non-null object

links = pd.read_csv('/Volumes/NO NAME/TMC_WalmartTest/data/links.csv', header = 0)
print(links.info())
links.head()
#9742 entries
#3 fileds - 
    #movieId    9742 non-null int64 
    #imdbId     9742 non-null int64 
    #tmdbId     9734 non-null float64

ratings = pd.read_csv('/Volumes/NO NAME/TMC_WalmartTest/data/ratings.csv', header = 0)
print(ratings.info())
ratings.head()
#100836 entries
#4 fileds - 
    #userId       100836 non-null int64
    #movieId      100836 non-null int64
    #rating       100836 non-null float64
    #timestamp    100836 non-null int64
    
tags = pd.read_csv('/Volumes/NO NAME/TMC_WalmartTest/data/tags.csv', header = 0)
print(tags.info())
tags.head()
#3683 entries
#4 fileds - 
    #userId       3683 non-null int64
    #movieId      3683 non-null int64
    #tag          3683 non-null object
    #timestamp    3683 non-null int64


#NOTES: for this exercise we will ignore links and tag data and build our hypothesis on movieId and Ratings
#will need to format UTC timestamp to date time 
    
#######################################DATA FORMAT#############################
#join data tables
join =  pd.merge(movies, ratings, on=["movieId"])
#convert UTC to readable time format
join['datetime'] = pd.to_datetime(join['timestamp'],unit='s')
join.head()

#i can not seem to get this funtion to work
#count= join.['genres'].str.count('|').add(1)


#extract movie date from title field and format as int
join['movie_year']  = join.title.apply(lambda st: st[st.find("(")+1:st.find(")")])
join['movie_year'] =join['movie_year'].str.replace('^[^\d]*', '')
join['movie_year']  = join['movie_year'] .str.strip()
join = join[join.movie_year != '']
join = join[join.movie_year.map(len) ==4]
join.dropna(inplace=True)
#convert to integer
join['movie_year'] = join['movie_year'].astype(int)
#extract year from ratings datetime
join['review_year'] = pd.DatetimeIndex(join['datetime']).year
#create timelapse variable that calculates years between movie release and year rated
join['timelapse']= join['review_year']-join['movie_year']

data=  join.drop(['timestamp','movieId','userId','genres','datetime'], axis=1)
data.head(20)

#lets look at some trend and corr plots
features= [ 'timelapse', 'movie_year', 'review_year']
for f in features:
    ax=plt.subplots(figsize=(6,3))
    ax=sb.regplot(x=data[f], y=data['rating'])
    plt.show()
#look at the corr matrix
sb.heatmap(data.corr())
#review_year/movie_year high pos corr
#rating/movie low neg corr
#rating/timelapse low pos corr
 
#Look at distribution of variables
sb.distplot(data.rating);
sb.distplot(data.timelapse);
sb.distplot(data.movie_year);
sb.distplot(data.review_year);

#Scatter plots
#look at relation between rating and timelapse
plt.scatter(data.rating, data.timelapse, alpha=0.5)
plt.show()  
#look at relation between rating and movie_year
plt.scatter(data.rating, data.movie_year, alpha=0.5)
plt.show() 

#Since rating is categorical i want to visualize the relationship via a violin plot
#rating and timelapse
    ax = plt.subplots(figsize=(7, 2.5))
    plt.xticks(rotation='vertical')
    ax=sb.violinplot(x="rating", y="timelapse", data=data, linewidth=1)
    plt.show() 

#rating and movie_year
        ax = plt.subplots(figsize=(7, 2.5))
    plt.xticks(rotation='vertical')
    ax=sb.violinplot(x="rating", y="movie_year", data=data, linewidth=1)
    plt.show() 
################################SIMPLE HYPOTHESIS TEST#########################

# QUESTION: Does the amount of time between when a movie is released and when someone rates it have any affect on it's rating score?

#HYPOTHESIS: I suspect there is a correlation between timelapse and rating
    #timelapse=number of years between movie release and review date
    #rating=1-5 value on likert scale (5 best 1 the worst)
    
    #H0: the two samples are independent (ρ0 = 0)
    #H1: there is a dependency between the samples ()
    
#TESTING: Correlation Tests - Pearson’s Correlation Coefficient
        #Tests whether two samples have a linear relationship

#ASSUMPTIONS:
        #Observations in each sample are independent and identically distributed (iid).
        #Observations in each sample are normally distributed.
        #Observations in each sample have the same variance.
        
#Lets first look at the covariance    
        
       
from numpy import cov
covariance = cov(data.timelapse, data.rating)
print(covariance)
        #[[188.28243217   1.16595741]
        #[  1.16595741   1.09179496]]
#cov = 1.165 - positive

#Next lets use Pearson's correlation coefficient to test strength of relationship (two tailed)
#Pearson's correlation coefficient = covariance(X, Y) / (stdv(X) * stdv(Y))
        #The coefficient returns a value between -1 and 1 that represents the limits of correlation from a full negative correlation to a full positive correlation.
        #High degree: If the coefficient value lies between ± 0.50 and ± 1, then it is said to be a strong correlation.
        #Moderate degree: If the value lies between ± 0.30 and ± 0.49, then it is said to be a medium correlation.
        #Low degree: When the value lies below + .29, then it is said to be a small correlation.
        #No correlation: When the value is zero.

from scipy.stats import pearsonr
corr, p = pearsonr(data.timelapse, data.rating)
print('Pearsons correlation: %.3f' % corr)
#Pearsons correlation: 0.081 - very little evidence of any correlation/ low correlation

#P VALUE METHOD
print(p)
#p=1.0368716002225904e-137

#1.0368716002225904e-137<0.05 - REJECT NULL

#TESTING SIGN OF CORR PARAMETER
    #significance: alpha=.05
    
#T STAT METHOD
    df = (data.shape[0])-2
    df
    #calculate test statistic
    t=((corr)*(df**.5))/((1-(corr**2))**.5)
    
#t=25.02045117018576
#crit values for a two tail @ aplha=.05 = +-1.96
  
  #25.02045117018576>1.96 - REJECT NULL  
#########################################RESULTS##############################
    
#In both testing methods we reject the null hypothesis that the population correlation is null 
  #Accept  alternative hypothesis that there is correlation between timelapse and rating
    
#There



