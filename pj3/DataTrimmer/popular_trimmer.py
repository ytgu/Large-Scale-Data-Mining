#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 15:48:54 2018

@author: AndrewXu
"""

import numpy as np
import collections
import csv
from numpy import genfromtxt

ratings_raw = genfromtxt('ratings.csv', delimiter=',', skip_header = 1)
ratings_raw_x, ratings_raw_y = ratings_raw.shape
movie_id_counter = collections.Counter(ratings_raw[:,1])
user_rating_counter = collections.Counter(ratings_raw[:,0])


##Popular movie trimming################################################################
popular_file = open('popular_trimmed.csv', 'w')
popular_movie_counter = movie_id_counter
to_delete = []
for movieID, movieFreq in popular_movie_counter.items():
    if (movieFreq < 3):
        to_delete.append(movieID)

for movieID in to_delete:        
    del popular_movie_counter[movieID]

writer = csv.writer(popular_file,lineterminator='\n')
for i in range(0,ratings_raw_x):
    if ratings_raw[i,1] in popular_movie_counter:
        writer.writerows([ratings_raw[i,:]])
        
popular_file.close()
