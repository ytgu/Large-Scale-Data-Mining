# -*- coding: utf-8 -*-

#PART 4

import os
import numpy as np
import matplotlib.pyplot as plt
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
from surprise import KNNWithMeans
from surprise import accuracy
from surprise.model_selection import KFold
import collections
from surprise.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from numpy import genfromtxt



#Q10
file_path = os.path.expanduser('./ratings.csv')
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines = 1)
data = Dataset.load_from_file(file_path, reader=reader)

rmse_mean = []
mae_mean = []

sim_options = {'name': 'pearson', 'user_based': True}
for i in range(2,102,2):
    algo = KNNWithMeans(k = i, sim_options=sim_options)
    result = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=10, verbose = True)
    rmse_mean.append(np.mean(result['test_rmse']))
    mae_mean.append(np.mean(result['test_mae']))

rmse_mean_array = np.asarray(rmse_mean)
mae_mean_array = np.asarray(mae_mean)

print('============================================')
print('Question 10:')
fig = plt.figure()
plt.plot(range(2, 102, 2),rmse_mean,'b-')
plt.xlabel('K Value')
plt.ylabel('RMSE')
plt.show()
fig.savefig('Part4_Q10_RMSE.png')

fig = plt.figure()
plt.plot(range(2, 102, 2), mae_mean,'b-')
plt.xlabel('K Value')
plt.ylabel('MAE')
plt.show()
fig.savefig('Part4_Q10_MAE.png')
print('============================================')



#Q11
print('============================================')
print('Question 11:')
min_k = 100
for i in range(len(rmse_mean_array)-2):
    index = (i+1)*2
    difference1 = np.abs(rmse_mean_array[i] - rmse_mean_array[i+1])
    difference2 = np.abs(rmse_mean_array[i+1] - rmse_mean_array[i+2])
    if((difference1<0.001) & (difference2<0.001)):
        if(index < min_k):
            min_k = index
            min_value = rmse_mean_array[i]
            
print('minimum K value for RMSE is %d' %min_k)
print('steady state RMSE is %f' %min_value)

min_k = 100
for i in range(len(mae_mean_array)-2):
    index = (i+1)*2
    difference1 = np.abs(mae_mean_array[i] - mae_mean_array[i+1])
    difference2 = np.abs(mae_mean_array[i+1] - mae_mean_array[i+2])
    if((difference1<0.001) & (difference2<0.001)):
        if(index < min_k):
            min_k = index
            min_value = mae_mean_array[i]

print('minimum K value for MAE is %d' %min_k)
print('steady state MAE is %f' %min_value)
print('============================================')




#Q12
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
sim_options = {'name': 'pearson', 'user_based': True}

for i in range(2,102,2):
    print("%d---------------------------------" % i)
    algo = KNNWithMeans(k = i, sim_options=sim_options)
    
    for j in range(0,len(total_train)):

        algo.fit(total_train[j])
            
        predictions = algo.test(total_trimmed[j])

        # Compute and print Root Mean Squared Error
        mean = []
        mean.append(accuracy.rmse(predictions, verbose=True))
    rmse_mean.append(np.mean(mean))
    
rmse_mean_array = np.asarray(rmse_mean)

print('============================================')
print('Question 12:')
fig = plt.figure()
plt.plot(range(2, 102, 2),rmse_mean,'b-')
plt.xlabel('K Value')
plt.ylabel('RMSE')
plt.show()
fig.savefig('Part4_Q12_RMSE.png')

index = np.where(rmse_mean_array == rmse_mean_array.min())
print('K Value for minimum RMSE is %d' % ((index[0]+1)*2))
print('minimum RMSE is %f' %rmse_mean_array.min())
print('============================================')




#Q13
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

sim_options = {'name': 'pearson', 'user_based': True}
for i in range(2,102,2):
    print("%d---------------------------------" % i)
    algo = KNNWithMeans(k = i, sim_options=sim_options)
    
    for j in range(0,len(total_train)):

        algo.fit(total_train[j])
            
        predictions = algo.test(total_trimmed[j])

        # Compute and print Root Mean Squared Error
        mean = []
        mean.append(accuracy.rmse(predictions, verbose=True))
    rmse_mean.append(np.mean(mean))
    
rmse_mean_array = np.asarray(rmse_mean)

print('============================================')
print('Question 13:')
fig = plt.figure()
plt.plot(range(2, 102, 2),rmse_mean,'b-')
plt.xlabel('K Value')
plt.ylabel('RMSE')
plt.show()
fig.savefig('Part4_Q13_RMSE.png')

index = np.where(rmse_mean_array == rmse_mean_array.min())
print('K Value for minimum RMSE is %d' % ((index[0]+1)*2))
print('minimum RMSE is %f' %rmse_mean_array.min())
print('============================================')





#Q14
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
sim_options = {'name': 'pearson', 'user_based': True}

for i in range(2,102,2):
    print("%d---------------------------------" % i)
    algo = KNNWithMeans(k = i, sim_options=sim_options)
    
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





#Q15
min_k_rmse = 24

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
    fig.savefig('Question15_thre_%d_%d.png' % (thres%10, (thres*10) % 10))

    
for thres in [2.5, 3, 3.5, 4]:
    file_path = os.path.expanduser('ratings.csv')
    reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines = 1)
    data_raw = Dataset.load_from_file(file_path, reader=reader)
    train, test = train_test_split(data_raw, test_size=0.1)
    sim_options = {'name': 'pearson', 'user_based': True}
    algo = KNNWithMeans(k = min_k_rmse, sim_options=sim_options)
    algo.fit(train)

    predictions = algo.test(test)

    target = []
    for i in range(len(test)):
        target.append(test[i][2])

    for t in range(len(target)):
        if target[t] > 2.5:
            target[t]=1
        else:
            target[t]=0

    predict = []
    for i in range(len(predictions)):
        predict.append(predictions[i][3])
        
    plot_roc(target, predict, thres)

