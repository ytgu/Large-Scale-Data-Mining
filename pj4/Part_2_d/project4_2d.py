
# coding: utf-8


#Question 2d
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


#part i
work_flow_0 = []
work_flow_1 = []
work_flow_2 = []
work_flow_3 = []
work_flow_4 = []
output_0 = []
output_1 = []
output_2 = []
output_3 = []
output_4 = []
count = 0
for k in features:
    if(k[3] == 0):
        work_flow_0.append(k)
        output_0.append(outputs[count])
    if(k[3] == 1):
        work_flow_1.append(k)
        output_1.append(output[count])
    if(k[3] == 2):
        work_flow_2.append(k)
        output_2.append(output[count])
    if(k[3] == 3):
        work_flow_3.append(k)
        output_3.append(output[count])
    if(k[3] == 4):
        work_flow_4.append(k)
        output_4.append(output[count])
    count = count+1

work_0 = np.asarray(work_flow_0)
work_1 = np.asarray(work_flow_1)
work_2 = np.asarray(work_flow_2)
work_3 = np.asarray(work_flow_3)
work_4 = np.asarray(work_flow_4)
out0 = np.asarray(output_0)
out1 = np.asarray(output_1)
out2 = np.asarray(output_2)
out3 = np.asarray(output_3)
out4 = np.asarray(output_4)

work_list = [work_0, work_1, work_2, work_3, work_4]
out_list = [out0, out1, out2, out3, out4]


#model training and testing using Linear Regression
clf = LinearRegression()#normalize=True

for i in range(5):
    train_rmse_list = []
    test_rmse_list = []
    w = work_list[i]
    out = out_list[i]
    for train_index, test_index in kf.split(w):
        clf.fit(w[train_index], out[train_index])
        train_rmse = rmse(out[train_index],clf.predict(w[train_index]))
        test_rmse = rmse(out[test_index],clf.predict(w[test_index]))
        train_rmse_list.append(train_rmse)
        test_rmse_list.append(test_rmse)
    predicted_values = clf.predict(w)
    true_values = out
    residues = predicted_values - true_values
    
    print('=============Plot of work flow %d ====================='%i)
    fig_i = plt.figure()
    plt.plot(true_values, predicted_values,'ro',linewidth=0.1)
    plt.plot([0,1],[0,1], '--', color='blue')
    plt.xlabel('True values')     
    plt.ylabel('Predicted values')
    plt.title('Predicted Values VS. True Values For Basic Linear Regression')
    plt.show()
    #fig_i.savefig(path+'fig/P2_i_ft.png')
    
    fig_i = plt.figure()
    plt.plot(predicted_values, residues,'ro',linewidth=0.1)
    plt.plot([0,1],[0,0], '--', color='blue')
    plt.xlabel('Predicted values')     
    plt.ylabel('Residual')
    plt.title('Residual VS. predicted Values For Basic Linear Regression')
    plt.show()
    #fig_i.savefig(path+'fig/P2_i_rt.png')
    
    print('train and test rmse of work flow %d'%i)
    print('train_rmse:')
    print(train_rmse_list)
    print('test_rmse:')
    print(test_rmse_list)
    print('=======================================================\n')






#Part ii
clf = LinearRegression()#normalize=True
for i in range(5):
    train_rmse_mean = []
    test_rmse_mean = []
    for j in range(10):
        train_rmse_list = []
        test_rmse_list = []
        w = work_list[i]
        out = out_list[i]
        for train_index, test_index in kf.split(w):
            poly = PolynomialFeatures(degree=j+1)
            w_temp = poly.fit_transform(w)
            clf.fit(w_temp[train_index], out[train_index])
            train_rmse = rmse(out[train_index],clf.predict(w_temp[train_index]))
            test_rmse = rmse(out[test_index],clf.predict(w_temp[test_index]))
            train_rmse_list.append(train_rmse)
            test_rmse_list.append(test_rmse)
        train_rmse_mean.append(mean_list(train_rmse_list))
        test_rmse_mean.append(mean_list(test_rmse_list))
        predicted_values = clf.predict(w_temp)
        true_values = out
          
        print('=======================================================')
        print('work flow %d'%i)
        print('train and test rmse of polynomial %d'%j)
        print('train_rmse:')
        print(train_rmse_list)
        print('test_rmse:')
        print(test_rmse_list)
        print('=======================================================\n')
    
    print('=============Plot of work flow %d ====================='%i)
    fig_ii = plt.figure()
    plt.plot(range(1,11), train_rmse_mean,'-',color= 'red', lw=1, label='Average training RMSE')
    plt.plot(range(1,11), test_rmse_mean,'-', color='blue', lw=1, label='Average testing RMSE')
    plt.xlabel('Index of polynomial')     
    plt.ylabel('Average RMSE')
    plt.legend(loc="upper right")
    plt.title('Average RMSE of Different Polynomial')
    plt.show()
    #fig_iv.savefig(path+'fig/P2_iv_ar.png')
    print('=======================================================\n')

