
# coding: utf-8

#Question 2e

import csv
import calendar
import re
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.preprocessing import PolynomialFeatures



def string_to_idx(s, reg):
    match = re.search(reg, s)
    return int(match.groups()[0])

def rmse(y, pred):
    return sqrt(mean_squared_error(y, pred))

def mean_list(l):
    return sum(l)/float(len(l))
#read data
path = '/Users/cindy/Documents/UCLA/ee219/project4/'
data = []
work_flow_list = []
file_name_list = []
with open(path+'network_backup_dataset.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        data.append((int(row['Week #']), row['Day of Week'],
                         int(row['Backup Start Time - Hour of Day']),row['Work-Flow-ID'],
                         row['File Name'],float(row['Size of Backup (GB)']),
                         int(row['Backup Time (hour)'])))
        work_flow_list.append(row['Work-Flow-ID'])
        file_name_list.append(row['File Name'])

work_flow_list = set(work_flow_list)
file_name_list = set(file_name_list)                     

#create dictionary                         
day_dict = {}
work_flow_ID_dict = {}
file_name_dict = {}
for idx in range(len(calendar.day_name)):
    day_dict[calendar.day_name[idx]] = idx+1

for w in work_flow_list:
    work_flow_ID_dict[w] = string_to_idx(w, '^work_flow_(\d+)')

for f in file_name_list:
    file_name_dict[f] = string_to_idx(f, '^File_(\d+)')

#feature vectorization
feature = []
output = []
""" week #, day of week, hour of day, workflow id, filename """
for datapoint in data:
    feature.append([datapoint[0], day_dict[datapoint[1]], datapoint[2],
                    work_flow_ID_dict[datapoint[3]], file_name_dict[datapoint[4]]])
    output.append(datapoint[5])
features = np.asarray(feature)
outputs = np.asarray(output)

#k-fold
num_fold = 10
kf = KFold(n_splits=num_fold)

train_rmse_mean = []
test_rmse_mean = []
for i in range(8):
    clf = KNeighborsRegressor(n_neighbors = i+1)
    train_rmse_list = []
    test_rmse_list = []
    for train_index, test_index in kf.split(features):
        clf.fit(features[train_index], outputs[train_index])
        train_rmse = rmse(outputs[train_index],clf.predict(features[train_index]))
        test_rmse = rmse(outputs[test_index],clf.predict(features[test_index]))
        train_rmse_list.append(train_rmse)
        test_rmse_list.append(test_rmse)
    predicted_values = clf.predict(features)
    true_values = outputs
    residues = predicted_values - true_values
    train_rmse_mean.append(mean_list(train_rmse_list))
    test_rmse_mean.append(mean_list(test_rmse_list))
    
    print('=============Plot of neighbors %d ====================='%(i+1))
    fig_i = plt.figure()
    plt.plot(true_values, predicted_values,'ro',linewidth=0.1)
    plt.plot([0,1],[0,1], '--', color='blue')
    plt.xlabel('True values')     
    plt.ylabel('Predicted values')
    plt.title('Predicted Values VS. True Values For KNN Regressor')
    plt.show()
    #fig_i.savefig(path+'fig/P2_i_ft.png')
    
    fig_i = plt.figure()
    plt.plot(predicted_values, residues,'ro',linewidth=0.1)
    plt.plot([0,1],[0,0], '--', color='blue')
    plt.xlabel('Predicted values')     
    plt.ylabel('Residual')
    plt.title('Residual VS. predicted Values For KNN Regressor')
    plt.show()
    #fig_i.savefig(path+'fig/P2_i_rt.png')
    
    print('train and test rmse of neighbors %d'%(i+1))
    print('train_rmse:')
    print(train_rmse_list)
    print('test_rmse:')
    print(test_rmse_list)
    print('=======================================================\n')
    
print('=======================================================')
fig_ii = plt.figure()
plt.plot(range(1,9), train_rmse_mean,'-',color= 'red', lw=1, label='Average training RMSE')
plt.plot(range(1,9), test_rmse_mean,'-', color='blue', lw=1, label='Average testing RMSE')
plt.xlabel('Index of neighbors')     
plt.ylabel('Average RMSE')
plt.legend(loc="lower right")
plt.title('Average RMSE of Different Neighbors')
plt.show()
#fig_iv.savefig(path+'fig/P2_iv_ar.png')
print('=======================================================\n')

