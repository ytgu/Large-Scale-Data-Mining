# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 18:20:20 2018

@author: Weiqian Xu
"""

import numpy as np
import collections
import csv
from numpy import genfromtxt

ratings_raw = genfromtxt('ratings.csv', delimiter=',', skip_header = 1)
ratings_raw_x, ratings_raw_y = ratings_raw.shape
movie_id_counter = collections.Counter(ratings_raw[:,1])
user_rating_counter = collections.Counter(ratings_raw[:,0])

#UnPopular movie trimming################################################################

unpopular_file = open('unpopular_trimmed.csv', 'w')
unpopular_movie_counter = movie_id_counter
to_delete2 = []
for movieID, movieFreq in unpopular_movie_counter.items():
    if (movieFreq > 2):
        to_delete2.append(movieID)
 
for movieID in to_delete2:
    del unpopular_movie_counter[movieID]

writer2 = csv.writer(unpopular_file,lineterminator='\n')
for i in range(0,ratings_raw_x):
    if ratings_raw[i,1] in unpopular_movie_counter:
        writer2.writerows([ratings_raw[i,:]])
 
unpopular_file.close()