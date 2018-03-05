# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 23:26:46 2018

@author: Weiqian Xu
"""

import csv
import re
import calendar
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt


file = open("log_logistic.txt", "w")

max_neurons = 200

def string_to_idx(s, reg):
    match = re.search(reg, s)
    return int(match.groups()[0])

def rmse(y, pred):
    return sqrt(mean_squared_error(y, pred))

data = []
work_flow_list = []
file_name_list = []
with open('network_backup_dataset.csv') as csvfile:
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
outputs = np.asarray(output,dtype=np.float64)

enc = OneHotEncoder()
feature_dict = {} 
for f in feature:
    for idx in range(len(f)):
        if idx not in feature_dict:
            feature_dict[idx] = [[f[idx]]]
        else:
            feature_dict[idx].append([f[idx]])

feature_ohc = {}
for k in feature_dict.keys():
    feature_ohc[k] = enc.fit_transform(feature_dict[k])

feature_ohc_list = []
for i in range(0, len(features)):
    temp_list = []
    for j in range(0, len(features[0])):
        temp_list += feature_ohc[j][i].toarray().tolist()[0]
    feature_ohc_list.append(temp_list)

feature_ohc_array = np.asarray(feature_ohc_list)
    
##Use Logistic activity function for nn
##sweep number of nuerons from 1 to 10
##plot Test-RMSE as function of number of hidden neurons
file.write("Logistic----------------------------------------------------\n")

num_fold = 10
kf = KFold(n_splits=num_fold)
total_train_rmse = []
total_test_rmse = []

for neuron_no in range(1, max_neurons+1):
    print ('training.....neurons=%d' % neuron_no)
    logistic_nn = MLPRegressor(hidden_layer_sizes = (neuron_no,), \
                            activation = 'logistic', verbose = False)
    train_rmse_list = []
    test_rmse_list = []
    for train_index, test_index in kf.split(feature_ohc_list):
        logistic_nn.fit(feature_ohc_array[train_index], outputs[train_index])
        train_rmse = rmse(outputs[train_index],logistic_nn.predict(feature_ohc_array[train_index]))
        test_rmse = rmse(outputs[test_index],logistic_nn.predict(feature_ohc_array[test_index]))
        train_rmse_list.append(train_rmse)
        test_rmse_list.append(test_rmse)
    total_train_rmse.append(np.mean(train_rmse_list))
    total_test_rmse.append(np.mean(test_rmse_list))

file.write("Train RMSE:")
for item in total_train_rmse:
   file.write('%s  ' % item)
file.write("\n\n")
file.write("Test RMSE:")
for item in total_test_rmse:
   file.write("%s  " % item)
file.write("\n\n")

fig = plt.figure()
plt.plot(range(1, max_neurons+1), total_test_rmse, 'b')
plt.xlabel('Number of hidden neurons (Activity Function = Logistic)')
plt.ylabel('Test RMSE')
plt.show()
fig.savefig('logistic.png')

total_train_rmse = np.asarray(total_train_rmse)
total_test_rmse = np.asarray(total_test_rmse)

best_neuron_no = np.where(total_train_rmse == total_train_rmse.min())[0] + 1
file.write("Best number of neuron is %d\n" % best_neuron_no)
file.write("The RMSE value related is %s\n" % total_train_rmse.min())
print ("Best number of neuron is %d" % best_neuron_no)
print ("The RMSE value related is %s" % total_train_rmse.min())


logistic_nn = MLPRegressor(hidden_layer_sizes = (best_neuron_no[0],), \
                            activation = 'logistic', verbose = False)
train_rmse_list = []
test_rmse_list = []
for train_index, test_index in kf.split(feature_ohc_list):
    logistic_nn.fit(feature_ohc_array[train_index], outputs[train_index])
    train_rmse = rmse(outputs[train_index],logistic_nn.predict(feature_ohc_array[train_index]))
    test_rmse = rmse(outputs[test_index],logistic_nn.predict(feature_ohc_array[test_index]))
    train_rmse_list.append(train_rmse)
    test_rmse_list.append(test_rmse)

fitted_values = logistic_nn.predict(feature_ohc_array)
true_values = outputs
residues = fitted_values - true_values

fig = plt.figure()
plt.plot(true_values, fitted_values,'ro',linewidth=0.1, markersize = 1)
plt.plot([0,1],[0,1], '--', color='blue')
plt.xlabel('True values')     
plt.ylabel('Fitted values')
plt.title('Fitted Values VS. True Values for Neural Network (logistic)')
plt.show()
fig.savefig('logistic_fitted.png')

fig = plt.figure()
plt.plot(true_values, residues,'ro',linewidth=0.1, markersize = 1)
plt.plot([0,1],[0,0], '--', color='blue')
plt.xlabel('True values')     
plt.ylabel('Residues')
plt.title('Residues VS. True Values for Neural Network (logistic)')
plt.show()
fig.savefig('logistic_res.png')


file.close()