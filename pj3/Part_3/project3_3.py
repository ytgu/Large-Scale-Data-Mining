# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 00:54:15 2018

@author: Weiqian Xu
"""

import numpy as np
import collections
import matplotlib.pyplot as plt
from numpy import genfromtxt

##Question 1################################################################
print ('Question 1-----------------------------------------------------\n')

ratings_raw = genfromtxt('ratings.csv', delimiter=',', skip_header = 1)
ratings_raw_x, ratings_raw_y = ratings_raw.shape
movie_id_counter = collections.Counter(ratings_raw[:,1])
user_rating_counter = collections.Counter(ratings_raw[:,0])

print ('minimum user number is: %d' % np.amin(ratings_raw, axis=0)[0])
print ('maximum user number is: %d' % np.amax(ratings_raw, axis=0)[0])
print ('minimum movie ID is: %d' % np.amin(ratings_raw, axis=0)[1])
print ('maximum movie ID is: %d\n' % np.amax(ratings_raw, axis=0)[1])
print ('total amount of samples is: %d' % ratings_raw_x)
print ('total number of moives is: %d' % len(movie_id_counter.values()))
print ('total number of users is: %d\n' % len(user_rating_counter.values()))

total_avail_ratings = ratings_raw_x
total_possi_ratings = len(movie_id_counter.values()) * len(user_rating_counter.values())
sparsity = total_avail_ratings/total_possi_ratings
print ('sparsity of data set is: %10.10f\n\n' % sparsity)

##Question 2################################################################
print ('Question 2-----------------------------------------------------\n')
fig = plt.figure()
plt.hist(ratings_raw[:,2], bins = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5], \
         align = 'left', rwidth = 0.5)
plt.title("Rating Distribution")
plt.xlabel("Ratings")
plt.ylabel("Frequency")
plt.show()
fig.savefig('Part3_2.png')

##Question 3################################################################
print ('Question 3-----------------------------------------------------\n')

movie_id_freq = sorted(movie_id_counter.values(), reverse=True)
fig = plt.figure()
plt.plot(movie_id_freq,'b-')
plt.ylabel('Frequency')
plt.xlabel('Movie index ordered by decreasing frequency')
plt.axes().get_xaxis().set_ticks([])
plt.show()
fig.savefig('Part3_q3.png')

##Question 4################################################################
print ('Question 4-----------------------------------------------------\n')
user_rating_freq = sorted(user_rating_counter.values(), reverse=True)
fig = plt.figure()
plt.plot(user_rating_freq,'b-')
plt.ylabel('Frequency')
plt.xlabel('User index ordered by decreasing frequency')
plt.axes().get_xaxis().set_ticks([])
plt.show()
fig.savefig('Part3_q4.png')

##Question 6################################################################
print ('Question 6-----------------------------------------------------\n')
counter = 0
max_movie_id = np.amax(ratings_raw, axis=0)[1].astype(np.int64)
ratings_variance = []

for i in range(0, max_movie_id+1):
    indices_of_movie = np.where(ratings_raw[:,1] == i)[0]
    if indices_of_movie.size:
        ratings = ratings_raw[indices_of_movie,2]
        ratings_variance.append(np.var(ratings))

print ("Maximum variance found is: %f" % np.amax(ratings_variance))
print ("Minimum variance found is: %f" % np.amin(ratings_variance))

fig = plt.figure()
plt.hist(ratings_variance, bins = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6], \
         align = 'mid')
plt.title("Variance Distribution")
plt.xlabel("Variance")
plt.ylabel("Frequency")
plt.show()
fig.savefig('Part3_q6.png')

    

















