# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 18:20:22 2018

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

##High Variance movie trimming#######################################################
high_variance_file = open('high_var_trimmed.csv', 'w')
high_variance_counter = movie_id_counter
to_delete3 = []
max_movie_id = np.amax(ratings_raw, axis=0)[1].astype(np.int64)

for i in range(0, max_movie_id+1):
    indices_of_movie = np.where(ratings_raw[:,1] == i)[0]
    if indices_of_movie.size:
        ratings = ratings_raw[indices_of_movie,2]
        if (np.var(ratings) < 2):
            del high_variance_counter[i]


for movieID, movieFreq in high_variance_counter.items():
    if (movieFreq < 5):
        to_delete3.append(movieID)

for movieID in to_delete3:
    del high_variance_counter[movieID]

writer3 = csv.writer(high_variance_file,lineterminator='\n')
for i in range(0,ratings_raw_x):
    if ratings_raw[i,1] in high_variance_counter:
        writer3.writerows([ratings_raw[i,:]])
high_variance_file.close()
