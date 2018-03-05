# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 17:15:36 2018

@author: ht
"""

import csv
import calendar
import re
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import f_regression, mutual_info_regression

path = '/users/ht/desktop/EE219/proj_4/'

def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

def string_to_idx(s, reg):
    match = re.search(reg, s)
    return int(match.groups()[0])

def rmse(y, pred):
    return sqrt(mean_squared_error(y, pred))

def mean_list(l):
    return sum(l)/float(len(l))
#read data
path = '/users/ht/desktop/EE219/proj_4/'
data = []
work_flow_list = []
file_name_list = []
with open(path+'data/network_backup_dataset.csv') as csvfile:
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

## i.
print '##### Part 2a.i #####'
#model training and testing
clf = LinearRegression()#normalize=True
train_rmse_list = []
test_rmse_list = []
for train_index, test_index in kf.split(features):
    clf.fit(features[train_index], outputs[train_index])
    train_rmse = rmse(outputs[train_index],clf.predict(features[train_index]))
    test_rmse = rmse(outputs[test_index],clf.predict(features[test_index]))
    train_rmse_list.append(train_rmse)
    test_rmse_list.append(test_rmse)
fitted_values = clf.predict(features)
true_values = outputs
residues = fitted_values - true_values
fig_i = plt.figure()
plt.plot(true_values, fitted_values,'ro',linewidth=0.1)
plt.plot([0,1],[0,1], '--', color='blue')
plt.xlabel('True values')     
plt.ylabel('Fitted values')
plt.title('Fitted Values VS. True Values For Basic Linear Regression')
plt.show()
fig_i.savefig(path+'fig/P2_i_ft.png')

fig_i = plt.figure()
plt.plot(fitted_values, residues,'ro',linewidth=0.1)
plt.plot([0,1],[0,0], '--', color='blue')
plt.xlabel('Fitted values')     
plt.ylabel('Residual')
plt.title('Residual VS. fitted Values For Basic Linear Regression')
plt.show()
fig_i.savefig(path+'fig/P2_i_rt.png')

print 'train_rmse:'
print train_rmse_list
print 'test_rmse:'
print test_rmse_list


## ii.
print '##### Part 2a.ii #####'
scaler = StandardScaler()
features_st = scaler.fit_transform(features)
clf = LinearRegression()#normalize=True
train_rmse_list_st = []
test_rmse_list_st = []
for train_index, test_index in kf.split(features_st):
    clf.fit(features_st[train_index], outputs[train_index])
    train_rmse = rmse(outputs[train_index],clf.predict(features_st[train_index]))
    test_rmse = rmse(outputs[test_index],clf.predict(features_st[test_index]))
    train_rmse_list_st.append(train_rmse)
    test_rmse_list_st.append(test_rmse)
fitted_values = clf.predict(features_st)
true_values = outputs
residues = fitted_values - true_values
fig_ii = plt.figure()
plt.plot(true_values, fitted_values,'ro',linewidth=0.1)
plt.plot([0,1],[0,1], '--', color='blue')
plt.xlabel('True values')     
plt.ylabel('Fitted values')
plt.title('Fitted Values VS. True Values For Standardized Linear Regression')
plt.show()
fig_ii.savefig(path+'fig/P2_ii_ft.png')

fig_ii = plt.figure()
plt.plot(fitted_values, residues,'ro',linewidth=0.1)
plt.plot([0,1],[0,0], '--', color='blue')
plt.xlabel('Fitted values')     
plt.ylabel('Residual')
plt.title('Residual VS. Fitted Values For Standardized Linear Regression')
plt.show()
fig_ii.savefig(path+'fig/P2_ii_rt.png')

print 'train_rmse:'
print train_rmse_list_st
print 'test_rmse:'
print test_rmse_list_st


## iii.
print '##### Part 2a.iii #####'
f_test, _ = f_regression(features, outputs)
mi = mutual_info_regression(features, outputs)
print 'f_test:'
print f_test
print 'mi:'
print mi

feature_sl = []
for f in feature:
    feature_sl.append([f[1],f[2],f[4]])
features_sl = np.asarray(feature_sl)

clf = LinearRegression()#normalize=True
train_rmse_list = []
test_rmse_list = []
for train_index, test_index in kf.split(features_sl):
    clf.fit(features_sl[train_index], outputs[train_index])
    train_rmse = rmse(outputs[train_index],clf.predict(features_sl[train_index]))
    test_rmse = rmse(outputs[test_index],clf.predict(features_sl[test_index]))
    train_rmse_list.append(train_rmse)
    test_rmse_list.append(test_rmse)
fitted_values = clf.predict(features_sl)
true_values = outputs
residues = fitted_values - true_values
fig_iii = plt.figure()
plt.plot(true_values, fitted_values,'ro',linewidth=0.1)
plt.plot([0,1],[0,1], '--', color='blue')
plt.xlabel('True values')     
plt.ylabel('Fitted values')
plt.title('Fitted Values VS. True Values For Basic Linear Regression')
plt.show()
fig_iii.savefig(path+'fig/P2_iii_ft.png')

fig_iii = plt.figure()
plt.plot(fitted_values, residues,'ro',linewidth=0.1)
plt.plot([0,1],[0,0], '--', color='blue')
plt.xlabel('Fitted values')     
plt.ylabel('Residual')
plt.title('Residual VS. Fitted Values For Basic Linear Regression')
plt.show()
fig_iii.savefig(path+'fig/P2_iii_rt.png')

print 'train_rmse:'
print train_rmse_list
print 'test_rmse:'
print test_rmse_list


## iv.
print '##### Part 2a.iv #####'
import itertools
com = ["".join(seq) for seq in itertools.product("01", repeat=5)]
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

#create 32 combinations of dataset
dataset = {}
for dec in com:
    dataset[dec] = []
    for idx_sample in range(len(feature)):
        temp = []
        for idx_feature in range(len(dec)):
            if dec[idx_feature] == '1':
                temp += feature_ohc[idx_feature][idx_sample].toarray().tolist()[0]
            else:
                temp.append(feature[idx_sample][idx_feature])
        dataset[dec].append(temp)

#train and test
train_armse_list = []
test_armse_list = []
for dec in com:
    clf = LinearRegression()
    features_com = np.asarray(dataset[dec])
    train_rmse_list = []
    test_rmse_list = []
    for train_index, test_index in kf.split(features_com):
        clf.fit(features_com[train_index], outputs[train_index])
        train_rmse = rmse(outputs[train_index],clf.predict(features_com[train_index]))
        test_rmse = rmse(outputs[test_index],clf.predict(features_com[test_index]))
        train_rmse_list.append(train_rmse)
        test_rmse_list.append(test_rmse)
    train_armse_list.append(mean_list(train_rmse_list))
    test_armse_list.append(mean_list(test_rmse_list))

fig_iv = plt.figure()
plt.plot(range(1,33), train_armse_list,'-',color= 'red', lw=1, label='Average training RMSE')
plt.plot(range(1,33), test_armse_list,'-', color='blue', lw=1, label='Average testing RMSE')
plt.xlabel('Index of combination of features')     
plt.ylabel('Average RMSE')
plt.title('Average RMSE of Different Combination of Features')
plt.show()
fig_iv.savefig(path+'fig/P2_iv_ar.png')

print 'Best combination in iv:'
print com[np.argmin(np.asarray(test_armse_list))]
print 'Min test avg RMSE:'
print min(test_armse_list)

## v.
print '##### Part 2a.v #####'
from sklearn.linear_model import Ridge, Lasso
train_armse_list_la = []
test_armse_list_la = []
for dec in com:
    clf = Lasso(alpha=0.001)
    features_com = np.asarray(dataset[dec])
    train_rmse_list = []
    test_rmse_list = []
    for train_index, test_index in kf.split(features_com):
        clf.fit(features_com[train_index], outputs[train_index])
        train_rmse = rmse(outputs[train_index],clf.predict(features_com[train_index]))
        test_rmse = rmse(outputs[test_index],clf.predict(features_com[test_index]))
        train_rmse_list.append(train_rmse)
        test_rmse_list.append(test_rmse)
    train_armse_list_la.append(mean_list(train_rmse_list))
    test_armse_list_la.append(mean_list(test_rmse_list))

fig_v = plt.figure()
plt.plot(range(1,33), train_armse_list_la,'-',color= 'red', lw=1, label='train')
plt.plot(range(1,33), test_armse_list_la,'-', color='blue', lw=1, label='test')
plt.xlabel('Index of combination of features')     
plt.ylabel('Average RMSE')
plt.title('Average RMSE of Different Combination of Features')
plt.legend(loc="upper right")
plt.show()
fig_v.savefig(path+'fig/P2_v_lasso.png')

#find min avg RMSE
idx_min = np.argmin(np.asarray(test_armse_list_la))
print 'Best combination in Lasso:'
print com[idx_min]

#find best parameter
from sklearn.model_selection import GridSearchCV
features_com = np.asarray(dataset[com[idx_min]])
parameters = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}
la = Lasso()
clf = GridSearchCV(la, parameters, scoring='neg_mean_squared_error') 
clf.fit(features_com, outputs)
print 'Best hyperparameter for Lasso:'
print clf.best_params_

#draw plot
clf = Lasso(alpha=0.001)
train_rmse_list = []
test_rmse_list = []
for train_index, test_index in kf.split(features_com):
    clf.fit(features_com[train_index], outputs[train_index])
    train_rmse = rmse(outputs[train_index],clf.predict(features_com[train_index]))
    test_rmse = rmse(outputs[test_index],clf.predict(features_com[test_index]))
    train_rmse_list.append(train_rmse)
    test_rmse_list.append(test_rmse)
fitted_values = clf.predict(features_com)
true_values = outputs
residues = fitted_values - true_values
fig_V = plt.figure()
plt.plot(true_values, fitted_values,'ro',linewidth=0.1)
plt.plot([0,1],[0,1], '--', color='blue')
plt.xlabel('True values')     
plt.ylabel('Fitted values')
plt.title('Fitted Values VS. True Values For Regularized Model Lasso')
plt.show()
fig_V.savefig(path+'fig/P2_v_la_ft.png')

fig_V = plt.figure()
plt.plot(fitted_values, residues,'ro',linewidth=0.1)
plt.plot([0,1],[0,0], '--', color='blue')
plt.xlabel('Fitted values')     
plt.ylabel('Residual')
plt.title('Residual VS. Fitted Values For Regularized Model Lasso')
plt.show()
fig_V.savefig(path+'fig/P2_v_la_rt.png')

print 'train_rmse:'
print train_rmse_list
print 'test_rmse:'
print test_rmse_list

train_armse_list_ri = []
test_armse_list_ri = []
for dec in com:
    clf = Ridge()#normalize=True
    features_com = np.asarray(dataset[dec])
    train_rmse_list = []
    test_rmse_list = []
    for train_index, test_index in kf.split(features_com):
        clf.fit(features_com[train_index], outputs[train_index])
        train_rmse = rmse(outputs[train_index],clf.predict(features_com[train_index]))
        test_rmse = rmse(outputs[test_index],clf.predict(features_com[test_index]))
        train_rmse_list.append(train_rmse)
        test_rmse_list.append(test_rmse)
    train_armse_list_ri.append(mean_list(train_rmse_list))
    test_armse_list_ri.append(mean_list(test_rmse_list))

fig_vr = plt.figure()
plt.plot(range(1,33), train_armse_list_ri,'-',color= 'red', lw=1, label='train')
plt.plot(range(1,33), test_armse_list_ri,'-', color='blue', lw=1, label='test')
plt.xlabel('Index of combination of features')     
plt.ylabel('Average RMSE')
plt.title('Average RMSE of Different Combination of Features')
plt.legend(loc="upper right")
plt.show()
fig_vr.savefig(path+'fig/P2_v_ridge.png')

#find min avg RMSE
idx_min = np.argmin(np.asarray(test_armse_list_ri))
print 'Best combination in Ridge:'
print com[idx_min]

#find best parameter
#from sklearn.model_selection import GridSearchCV
features_com = np.asarray(dataset[com[idx_min]])
parameters = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
ri = Ridge()
clf = GridSearchCV(ri, parameters, scoring='neg_mean_squared_error') 
clf.fit(features_com, outputs)
print 'Best hyperparameter in Ridge:'
print clf.best_params_

#draw plot
clf = Ridge(alpha=0.1)
train_rmse_list = []
test_rmse_list = []
for train_index, test_index in kf.split(features_com):
    clf.fit(features_com[train_index], outputs[train_index])
    train_rmse = rmse(outputs[train_index],clf.predict(features_com[train_index]))
    test_rmse = rmse(outputs[test_index],clf.predict(features_com[test_index]))
    train_rmse_list.append(train_rmse)
    test_rmse_list.append(test_rmse)
fitted_values = clf.predict(features_com)
true_values = outputs
residues = fitted_values - true_values
fig_V = plt.figure()
plt.plot(true_values, fitted_values,'ro',linewidth=0.1)
plt.plot([0,1],[0,1], '--', color='blue')
plt.xlabel('True values')     
plt.ylabel('Fitted values')
plt.title('Fitted Values VS. True Values For Regularized Model Ridge')
plt.show()
fig_V.savefig(path+'fig/P2_v_ri_ft.png')

fig_V = plt.figure()
plt.plot(fitted_values, residues,'ro',linewidth=0.1)
plt.plot([0,1],[0,0], '--', color='blue')
plt.xlabel('Fitted values')     
plt.ylabel('Residual')
plt.title('Residual VS. Fitted Values For Regularized Model Ridge')
plt.show()
fig_V.savefig(path+'fig/P2_v_ri_rt.png')

print 'train_rmse:'
print train_rmse_list
print 'test_rmse:'
print test_rmse_list





