# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 23:32:25 2018

@author: Weiqian Xu
"""


#PART 5.2

import os
import numpy as np
import matplotlib.pyplot as plt
from surprise import Dataset
from surprise import Reader
from surprise import NMF
from surprise.model_selection import cross_validate
from surprise import accuracy
from surprise.model_selection import KFold
import collections
from numpy import genfromtxt
from surprise.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc



##Question 17 & 18
file_path = os.path.expanduser('ratings.csv')
reader = Reader(line_format='user item rating', sep=',')
data = Dataset.load_from_file(file_path, reader=reader)

rmse_mean = []
mae_mean = []

for i in range(2,52,2):
    algo = NMF(n_factors = i, random_state = 1)
    result = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=10)
    rmse_mean.append(np.mean(result['test_rmse']))
    mae_mean.append(np.mean(result['test_mae']))

rmse_mean_array = np.asarray(rmse_mean)
mae_mean_array = np.asarray(mae_mean)

fig = plt.figure()
plt.plot(range(2, 52, 2),rmse_mean,'b-')
plt.xlabel('K Value')
plt.ylabel('RMSE')
plt.show()
fig.savefig('Part5_Q17_RMSE.png')

index = np.where(rmse_mean_array == rmse_mean_array.min())
print('K Value for minimum RMSE is %d' % ((index[0]+1)*2))

fig = plt.figure()
plt.plot(range(2, 52, 2), mae_mean,'b-')
plt.xlabel('K Value')
plt.ylabel('MAE')
plt.show()
fig.savefig('Part5_Q17_MAE.png')

index = np.where(mae_mean_array == mae_mean_array.min())
print('K Value for minimum MAE is %d' % ((index[0]+1)*2))






#Q19
file_path = os.path.expanduser('ratings.csv')
reader = Reader(line_format='user item rating', sep=',', skip_lines = 1)
data = Dataset.load_from_file(file_path, reader=reader)

kf = KFold(n_splits=10)

total_train = []
total_test = []
total_trimmed = []

for trainset, testset in kf.split(data):
    
    total_train.append(trainset)
    total_test.append(testset)
    
    movie_id = []
    to_delete=[]
    
    for user, movie, rating in testset:
        movie_id.append(movie)

    c = collections.Counter(movie_id)
    to_delete = []

for movieID, movieFreq in c.items():
    if(movieFreq <= 2):
        to_delete.append(movieID)
    
    trimmed_testset = []
    for i in range(len(testset)):
        if(testset[i][1] not in to_delete):
            trimmed_testset.append(testset[i])
    total_trimmed.append(trimmed_testset)

rmse_mean = []

for i in range(2,52,2):
    print("%d---------------------------------" % i)
    algo = NMF(n_factors = i, random_state = 1)
    
    for j in range(0,len(total_train)):
        
        algo.fit(total_train[j])
        
        predictions = algo.test(total_trimmed[j])
        
        # Compute and print Root Mean Squared Error
        mean = []
        mean.append(accuracy.rmse(predictions, verbose=True))
    rmse_mean.append(np.mean(mean))

rmse_mean_array = np.asarray(rmse_mean)

print('============================================')
print('Question 19:')
fig = plt.figure()
plt.plot(range(2, 52, 2),rmse_mean,'b-')
plt.xlabel('K Value')
plt.ylabel('RMSE')
plt.show()
fig.savefig('Part5_Q19_RMSE.png')

index = np.where(rmse_mean_array == rmse_mean_array.min())
print('K Value for minimum RMSE is %d' % ((index[0]+1)*2))

print('============================================')





#Q20
file_path = os.path.expanduser('ratings.csv')
reader = Reader(line_format='user item rating', sep=',', skip_lines = 1)
data = Dataset.load_from_file(file_path, reader=reader)

kf = KFold(n_splits=10)

total_train = []
total_test = []
total_trimmed = []

for trainset, testset in kf.split(data):
    
    total_train.append(trainset)
    total_test.append(testset)
    
    movie_id = []
    to_delete=[]
    
    for user, movie, rating in testset:
        movie_id.append(movie)

    c = collections.Counter(movie_id)
    to_delete = []

for movieID, movieFreq in c.items():
    if(movieFreq > 2):
        to_delete.append(movieID)
    
    trimmed_testset = []
    for i in range(len(testset)):
        if(testset[i][1] not in to_delete):
            trimmed_testset.append(testset[i])
    total_trimmed.append(trimmed_testset)

rmse_mean = []

for i in range(2,52,2):
    print("%d---------------------------------" % i)
    algo = NMF(n_factors = i, random_state = 1)
    
    for j in range(0,len(total_train)):
        
        algo.fit(total_train[j])
        
        predictions = algo.test(total_trimmed[j])
        
        # Compute and print Root Mean Squared Error
        mean = []
        mean.append(accuracy.rmse(predictions, verbose=True))
    rmse_mean.append(np.mean(mean))

rmse_mean_array = np.asarray(rmse_mean)

print('============================================')
print('Question 20:')
fig = plt.figure()
plt.plot(range(2, 52, 2),rmse_mean,'b-')
plt.xlabel('K Value')
plt.ylabel('RMSE')
plt.show()
fig.savefig('Part5_Q20_RMSE.png')

index = np.where(rmse_mean_array == rmse_mean_array.min())
print('K Value for minimum RMSE is %d' % ((index[0]+1)*2))

print('============================================')






#Q21
ratings_raw = genfromtxt('ratings.csv', delimiter=',', skip_header = 1)
ratings_raw_x, ratings_raw_y = ratings_raw.shape
movie_id_counter = collections.Counter(ratings_raw[:,1])
user_rating_counter = collections.Counter(ratings_raw[:,0])

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

file_path = os.path.expanduser('ratings.csv')
reader = Reader(line_format='user item rating', sep=',', skip_lines = 1)
data = Dataset.load_from_file(file_path, reader=reader)

kf = KFold(n_splits=10)

total_train = []
total_test = []
total_trimmed = []

for trainset, testset in kf.split(data):
    total_train.append(trainset)
    total_test.append(testset)
    trimmed_testset = []
    for i in range(len(testset)):
        if(int(testset[i][1]) in high_variance_counter):
            trimmed_testset.append(testset[i])
    total_trimmed.append(trimmed_testset)

rmse_mean = []

for i in range(2,52,2):
    print("%d---------------------------------" % i)
    algo = NMF(n_factors = i, random_state = 1)
    
    for j in range(0,len(total_train)):
        
        algo.fit(total_train[j])
        
        predictions = algo.test(total_trimmed[j])
        
        # Compute and print Root Mean Squared Error
        mean = []
        if not predictions:
            mean.append(0)
            print('RMSE: 0.0000')
        else:
            mean.append(accuracy.rmse(predictions, verbose=True))
    rmse_mean.append(np.mean(mean))

rmse_mean_array = np.asarray(rmse_mean)

print('============================================')
print('Question 21:')
fig = plt.figure()
plt.plot(range(2, 52, 2),rmse_mean,'b-')
plt.xlabel('K Value')
plt.ylabel('RMSE')
plt.show()
fig.savefig('Part5_Q21_RMSE.png')

index = np.where(rmse_mean_array == rmse_mean_array.min())
print('K Value for minimum RMSE is %d' % ((index[0]+1)*2))

print('============================================')





#Q22
k_min_rmse = 18

def plot_roc(original, predict, thres):
    fpr,tpr,thresholds = roc_curve(original, predict)
    auc_roc = auc(fpr,tpr)
    fig = plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate (AUC = %0.6f)' % auc_roc)
    plt.ylabel('True Positive Rate')
    plt.title('Threshold = %2.1f' % thres)
    plt.grid(color='0.7', linestyle='-', linewidth=1)
    plt.show()
    fig.savefig('Question22_thre_%d_%d.png' % (thres%10, (thres*10) % 10))


for thres in [2.5, 3, 3.5, 4]:
    file_path = os.path.expanduser('ratings.csv')
    reader = Reader(line_format='user item rating timestamp', sep=',')
    data_raw = Dataset.load_from_file(file_path, reader=reader)
    train, test = train_test_split(data_raw, test_size=0.1)
    algo = NMF(n_factors = k_min_rmse, random_state = 1)
    algo.fit(train)
    
    predictions = algo.test(test)
    
    target = []
    for i in range(len(test)):
        target.append(test[i][2])
    
    for t in range(len(target)):
        if target[t] > thres:
            target[t]=1
        else:
            target[t]=0

predict = []
for i in range(len(predictions)):
    predict.append(predictions[i][3])
    
    plot_roc(target, predict, thres)






#Q23
import csv
import numpy as np
from sklearn.decomposition import NMF
from scipy import sparse
path = '/users/ht/desktop/EE219/proj_3/data/'

rate_list = []
user = []
movie = []
with open(path+'ratings.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        rate_list.append((int(row['userId']), int(row['movieId']), float(row['rating'])))
        user.append(int(row['userId']))
        movie.append(int(row['movieId']))
size_user = max(user)
size_movie = max(movie)

with open(path+'movies.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    mvid_genr = dict([(int(row['movieId']),row['genres']) for row in reader])

#create ratings matrix
R = np.zeros((size_user, size_movie))
for rate in rate_list:
    R[rate[0]-1][rate[1]-1] = rate[2]

#NMF
R_s = sparse.csr_matrix(R)
nmf = NMF(n_components=20, init='random', random_state=0)
U = nmf.fit_transform(R)
V = (nmf.components_).T

top_10_list = []
for i in range(20):
    temp = np.argsort(V[:,i])
    temp += 1
    reverse = temp[::-1]
    top_10_list.append(reverse[:10])

top_10_genr_list = []
for id_ls in top_10_list:
    sub_list = []
    for id in id_ls:
        sub_list.append(mvid_genr[id])
    top_10_genr_list.append(sub_list)

print "top 10 list (sample 1):"
for genre in top_10_genr_list[0]:
    print genre
print ''
print "top 10 list (sample 2):"
for genre in top_10_genr_list[4]:
    print genre
print ''
print "top 10 list (sample 3):"
for genre in top_10_genr_list[7]:
    print genre
