
# coding: utf-8


#PART 6
import os
import numpy as np
import matplotlib.pyplot as plt
from surprise import Dataset
from surprise import Reader
from surprise import NMF
from surprise import SVD
from surprise.model_selection import cross_validate
from surprise import accuracy
from surprise.model_selection import KFold
import collections
from numpy import genfromtxt
from surprise.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt




#Q30
def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))

header = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('ratings.csv', sep=',', names=header, skiprows=1)
# print(df)
n_users = df.user_id.max()
n_items = df.item_id.max()

kf = KFold(n_splits=10, random_state=2)

result = []
for ds in kf.split(df):
    train_data, test_data = df.iloc[ds[0]], df.iloc[ds[1]]

    all_data_matrix = np.zeros((n_users, n_items))
    for line in df.itertuples():
        all_data_matrix[line[1]-1, line[2]-1] = line[3]

    test_data_matrix = np.zeros((n_users, n_items))
    for line in test_data.itertuples():
        test_data_matrix[line[1]-1, line[2]-1] = line[3]

    predict_data_matrix = np.zeros((n_users, n_items))
    for i in range(n_users):
        temp = np.sum(all_data_matrix[i, :])
        if(temp != 0):
            temp = temp / np.count_nonzero(all_data_matrix[i, :])
        predict_data_matrix[i, np.where(test_data_matrix[i, :]!=0)] = temp 
    
    result.append(rmse(predict_data_matrix, test_data_matrix))

print('Average RMSE is: ', np.mean(result))





#Q31
def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))

header = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('popular_trimmed.csv', sep=',', names=header)
# print(df)
n_users = int(df.user_id.max())
n_items = int(df.item_id.max())

kf = KFold(n_splits=10, random_state=2)

result = []
for ds in kf.split(df):
    train_data, test_data = df.iloc[ds[0]], df.iloc[ds[1]]

    all_data_matrix = np.zeros((n_users, n_items))
    for line in df.itertuples():
        all_data_matrix[int(line[1]-1), int(line[2]-1)] = line[3]

    test_data_matrix = np.zeros((n_users, n_items))
    for line in test_data.itertuples():
        test_data_matrix[int(line[1]-1), int(line[2]-1)] = line[3]

    predict_data_matrix = np.zeros((n_users, n_items))
    for i in range(n_users):
        temp = np.sum(all_data_matrix[i, :])
        if(temp != 0):
            temp = temp / np.count_nonzero(all_data_matrix[i, :])
        predict_data_matrix[i, np.where(test_data_matrix[i, :]!=0)] = temp 
    
    result.append(rmse(predict_data_matrix, test_data_matrix))

print('Average RMSE is: ', np.mean(result))





#Q32
def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))

header = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('unpopular_trimmed.csv', sep=',', names=header)
# print(df)
n_users = int(df.user_id.max())
n_items = int(df.item_id.max())

kf = KFold(n_splits=10, random_state=2)

result = []
for ds in kf.split(df):
    train_data, test_data = df.iloc[ds[0]], df.iloc[ds[1]]

    all_data_matrix = np.zeros((n_users, n_items))
    for line in df.itertuples():
        all_data_matrix[int(line[1]-1), int(line[2]-1)] = line[3]

    test_data_matrix = np.zeros((n_users, n_items))
    for line in test_data.itertuples():
        test_data_matrix[int(line[1]-1), int(line[2]-1)] = line[3]

    predict_data_matrix = np.zeros((n_users, n_items))
    for i in range(n_users):
        temp = np.sum(all_data_matrix[i, :])
        if(temp != 0):
            temp = temp / np.count_nonzero(all_data_matrix[i, :])
        predict_data_matrix[i, np.where(test_data_matrix[i, :]!=0)] = temp 
    
    result.append(rmse(predict_data_matrix, test_data_matrix))

print('Average RMSE is: ', np.mean(result))





#Q33
def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))

header = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('high_var_trimmed.csv', sep=',', names=header)
# print(df)
n_users = int(df.user_id.max())
n_items = int(df.item_id.max())

kf = KFold(n_splits=10, random_state=2)

result = []
for ds in kf.split(df):
    train_data, test_data = df.iloc[ds[0]], df.iloc[ds[1]]

    all_data_matrix = np.zeros((n_users, n_items))
    for line in df.itertuples():
        all_data_matrix[int(line[1]-1), int(line[2]-1)] = line[3]

    test_data_matrix = np.zeros((n_users, n_items))
    for line in test_data.itertuples():
        test_data_matrix[int(line[1]-1), int(line[2]-1)] = line[3]

    predict_data_matrix = np.zeros((n_users, n_items))
    for i in range(n_users):
        temp = np.sum(all_data_matrix[i, :])
        if(temp != 0):
            temp = temp / np.count_nonzero(all_data_matrix[i, :])
        predict_data_matrix[i, np.where(test_data_matrix[i, :]!=0)] = temp 
    
    result.append(rmse(predict_data_matrix, test_data_matrix))

print('Average RMSE is: ', np.mean(result))





#PART 7


#Q34
def plot_roc_mul(original1, predict1, original2, predict2, original3, predict3, thres):
    fig = plt.figure()
    fpr1,tpr1,thresholds1 = roc_curve(original1, predict1)  
    plt.plot(fpr1, tpr1)
    fpr2,tpr2,thresholds2 = roc_curve(original2, predict2)  
    plt.plot(fpr2, tpr2)
    fpr3,tpr3,thresholds3 = roc_curve(original3, predict3)  
    plt.plot(fpr3, tpr3)
    plt.legend(['KNN', 'NNMF', 'MF with Bias'])
    plt.xlabel('False Positive Rate')     
    plt.ylabel('True Positive Rate')
    plt.title('Threshold = %2.1f' % thres)
    plt.grid(color='0.7', linestyle='-', linewidth=1)
    plt.show()
    fig.savefig('Question34_thre_%d_%d.png' % (thres%10, (thres*10) % 10))


thres = 3
file_path = os.path.expanduser('ratings.csv')
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
data_raw = Dataset.load_from_file(file_path, reader=reader)
train, test = train_test_split(data_raw, test_size=0.1)

sim_options = {'name': 'pearson', 'user_based': True}
algo1 = KNNWithMeans(k = 24, sim_options=sim_options) # !!!!!! need to revise after part 4 done
algo2 = NMF(n_factors = 18, random_state = 1)
algo3 = SVD(n_factors = 16, random_state = 1, biased=True)
algo1.fit(train)
algo2.fit(train)
algo3.fit(train)

predictions1 = algo1.test(test)
predictions2 = algo2.test(test)
predictions3 = algo3.test(test)

target1 = []
for i in range(len(test)):
    target1.append(test[i][2])

for t in range(len(target1)):
    if target1[t] > thres:
        target1[t]=1
    else:
        target1[t]=0

target2 = []
for i in range(len(test)):
    target2.append(test[i][2])

for t in range(len(target2)):
    if target2[t] > thres:
        target2[t]=1
    else:
        target2[t]=0
        
target3 = []
for i in range(len(test)):
    target3.append(test[i][2])

for t in range(len(target3)):
    if target3[t] > thres:
        target3[t]=1
    else:
        target3[t]=0

predict1 = []
for i in range(len(predictions1)):
    predict1.append(predictions1[i][3])

predict2 = []
for i in range(len(predictions2)):
    predict2.append(predictions2[i][3])
    
predict3 = []
for i in range(len(predictions3)):
    predict3.append(predictions3[i][3])

plot_roc_mul(target1, predict1, target2, predict2, target3, predict3, thres)

