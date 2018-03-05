import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('network_backup_dataset.csv', ',')
df_re = df.replace({'Day of Week': {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7},
                  'Work-Flow-ID': {'work_flow_0': 0, 'work_flow_1': 1, 'work_flow_2': 2, 'work_flow_3': 3, 'work_flow_4': 4},
                  'File Name': {'File_0': 0, 'File_1': 1, 'File_2': 2, 'File_3': 3, 'File_4': 4, 'File_5': 5,'File_6': 6, 'File_7': 7, 
                                'File_8': 8, 'File_9': 9, 'File_10': 10, 'File_11': 11, 'File_12': 12, 'File_13': 13, 'File_14': 14, 
                                'File_15': 15, 'File_16': 16, 'File_17': 17, 'File_18': 18, 'File_19': 19, 'File_20': 20, 'File_21': 21,
                                'File_22': 22, 'File_23': 23, 'File_24': 24, 'File_25': 25, 'File_26': 26, 'File_27': 27, 'File_28': 28, 'File_29': 29, }})
X_o = df_re[['Week #', 'Day of Week', 'Backup Start Time - Hour of Day', 'Work-Flow-ID', 'File Name']]
y_o = df_re['Size of Backup (GB)']
X = np.asarray(X_o)
y = np.asarray(y_o)

num_fold = 10
kf = KFold(n_splits=num_fold, shuffle=True, random_state=0)
best_num_of_trees = 20
best_max_depth = 4
best_max_features = 5

clf = RandomForestRegressor(n_estimators=best_num_of_trees, max_depth=best_max_depth, random_state=2,
                                      max_features=best_max_features, oob_score=True)
train_mse_list = []
test_mse_list = []
# ooberror_list = []
for train_index, test_index in kf.split(X):
    clf.fit(X[train_index], y[train_index])
    train_mse = mean_squared_error(y[train_index], clf.predict(X[train_index]))
    test_mse = mean_squared_error(y[test_index], clf.predict(X[test_index]))
#     ooberror = 1 - clf.oob_score_
    
    train_mse_list.append(train_mse)
    test_mse_list.append(test_mse)
#     ooberror_list.append(ooberror)

clf.fit(X, y)
fitted_values = clf.predict(X)
true_values = y
residues = true_values - fitted_values
ooberror = 1 - clf.oob_score_

print ('train_rmse:', np.sqrt(train_mse_list))
print ('test_rmse:', np.sqrt(test_mse_list))
print ('average_train_rmse:', np.sqrt(np.mean(train_mse_list)))
print ('average_test_rmse:', np.sqrt(np.mean(test_mse_list)))
print ('oob_error:', ooberror)

fig = plt.figure()
plt.scatter(range(0, len(y)), true_values, s=0.2)
plt.scatter(range(0, len(y)), fitted_values, s=0.2)
plt.legend(('true', 'fitted'))
plt.xlabel('Data')     
plt.ylabel('Values')
plt.title('Fitted Values VS. True Values For Random Forest Regression')
plt.show()
fig.savefig('2b11')

fig = plt.figure()
plt.scatter(range(0, len(y)), residues, s=0.2)
plt.scatter(range(0, len(y)), fitted_values, s=0.2)
plt.legend(('residues', 'fitted'))
plt.xlabel('Data')     
plt.ylabel('Values')
plt.title('Residues VS. Fitted Values For Random Forest Regression')
plt.show()
fig.savefig('2b12')

ooberror_list = []
fig = plt.figure()
for j in range(1, 6, 1):
    ooberror_list = []
    for i in range(1, 201, 1):
        clf = RandomForestRegressor(n_estimators=i, max_depth=best_max_depth,
                                          max_features=j, random_state=2, oob_score=True)
        clf.fit(X, y)
        ooberror = 1 - clf.oob_score_
        ooberror_list.append(ooberror)
        
    print('best number of trees for %d =' %j, np.argmin(ooberror_list)+1)
    print('the best oob error is ', np.min(ooberror_list))
    plt.plot(range(1, 201, 1), ooberror_list, label=j)
    # plt.legend(('true', 'fitted'))
    plt.xlabel('number of trees')     
    plt.ylabel('oob error')
    plt.title('oob error VS. number of trees For Random Forest Regression')
plt.legend()
plt.show()
fig.savefig('2b21')

fig = plt.figure()
for j in range(1, 6, 1):
    sweep = {'n_estimators': range(1, 201, 1)}
    clf = RandomForestRegressor(max_depth=best_max_depth, max_features=j, random_state=2)
    cvresult = GridSearchCV(estimator=clf, param_grid=sweep, scoring='neg_mean_squared_error', cv=10)
    cvresult.fit(X, y)
    tempre = cvresult.cv_results_
    rmse_full = np.sqrt(abs(tempre['mean_test_score']))
    best_RMSE = np.sqrt(abs(cvresult.best_score_))
    best_number_tree = cvresult.best_params_['n_estimators']
    
    print('best number of trees for %d =' %j, best_number_tree)
    print('the best average test rmse is ', best_RMSE)

    plt.plot(range(1, 201, 1), rmse_full, label=j)
    plt.xlabel('number of trees')     
    plt.ylabel('Average Test RMSE')
    plt.title('Average Test RMSE VS. number of trees For Random Forest Regression')
    
plt.legend(loc='upper right')
plt.show()
fig.savefig('2b22')

num_fold = 10
kf = KFold(n_splits=num_fold, shuffle=True, random_state=0)
best_num_of_trees = 156
best_max_depth = 4
best_max_features = 3

clf = RandomForestRegressor(n_estimators=best_num_of_trees, max_depth=best_max_depth, random_state=2,
                                      max_features=best_max_features, oob_score=True)
train_mse_list = []
test_mse_list = []
# ooberror_list = []
for train_index, test_index in kf.split(X):
    clf.fit(X[train_index], y[train_index])
    train_mse = mean_squared_error(y[train_index], clf.predict(X[train_index]))
    test_mse = mean_squared_error(y[test_index], clf.predict(X[test_index]))
#     ooberror = 1 - clf.oob_score_
    
    train_mse_list.append(train_mse)
    test_mse_list.append(test_mse)
#     ooberror_list.append(ooberror)

clf.fit(X, y)
fitted_values = clf.predict(X)
true_values = y
residues = true_values - fitted_values
ooberror = 1 - clf.oob_score_

print ('train_rmse:', np.sqrt(train_mse_list))
print ('test_rmse:', np.sqrt(test_mse_list))
print ('average_train_rmse:', np.sqrt(np.mean(train_mse_list)))
print ('average_test_rmse:', np.sqrt(np.mean(test_mse_list)))
print ('oob_error:', ooberror)

fig = plt.figure()
plt.scatter(range(0, len(y)), true_values, s=0.2)
plt.scatter(range(0, len(y)), fitted_values, s=0.2)
plt.legend(('true', 'fitted'))
plt.xlabel('Data')     
plt.ylabel('Values')
plt.title('Fitted Values VS. True Values For Random Forest Regression')
plt.show()
fig.savefig('2b23')

fig = plt.figure()
plt.scatter(range(0, len(y)), residues, s=0.2)
plt.scatter(range(0, len(y)), fitted_values, s=0.2)
plt.legend(('residues', 'fitted'))
plt.xlabel('Data')     
plt.ylabel('Values')
plt.title('Residues VS. Fitted Values For Random Forest Regression')
plt.show()
fig.savefig('2b24')

ooberror_list = []
fig = plt.figure()
for i in range(1, 16, 1):
    clf = RandomForestRegressor(n_estimators=156, max_depth=i,
                                      max_features=3, random_state=2, oob_score=True)
    clf.fit(X, y)
    ooberror = 1 - clf.oob_score_
    ooberror_list.append(ooberror)

print('best max depth =', np.argmin(ooberror_list)+1)
print('the best oob error is ', np.min(ooberror_list))
#     print(ooberror_list)
plt.plot(range(1, 16, 1), ooberror_list)
# plt.legend(('true', 'fitted'))
plt.xlabel('max depth')     
plt.ylabel('oob error')
plt.title('oob error VS. max depth For Random Forest Regression')
plt.legend()
plt.show()
fig.savefig('2b31')

fig = plt.figure()
sweep = {'max_depth': range(1, 16, 1)}
clf = RandomForestRegressor(n_estimators=156, max_features=3, random_state=2)
cvresult = GridSearchCV(estimator=clf, param_grid=sweep, scoring='neg_mean_squared_error', cv=10)
cvresult.fit(X, y)
tempre = cvresult.cv_results_
rmse_full = np.sqrt(abs(tempre['mean_test_score']))
best_RMSE = np.sqrt(abs(cvresult.best_score_))
best_max_depth = cvresult.best_params_['max_depth']

print('best max depth for =', best_max_depth)
print('the best average test rmse is ', best_RMSE)

plt.plot(range(1, 16, 1), rmse_full)
plt.xlabel('max depth')     
plt.ylabel('Average Test RMSE')
plt.title('Average Test RMSE VS. max depth For Random Forest Regression')
    
plt.legend()
plt.show()
fig.savefig('2b32')

num_fold = 10
kf = KFold(n_splits=num_fold, shuffle=True, random_state=0)
best_num_of_trees = 156
best_max_depth = 10
best_max_features = 3

clf = RandomForestRegressor(n_estimators=best_num_of_trees, max_depth=best_max_depth, random_state=2,
                                      max_features=best_max_features, oob_score=True)
train_mse_list = []
test_mse_list = []
# ooberror_list = []
for train_index, test_index in kf.split(X):
    clf.fit(X[train_index], y[train_index])
    train_mse = mean_squared_error(y[train_index], clf.predict(X[train_index]))
    test_mse = mean_squared_error(y[test_index], clf.predict(X[test_index]))
#     ooberror = 1 - clf.oob_score_
    
    train_mse_list.append(train_mse)
    test_mse_list.append(test_mse)
#     ooberror_list.append(ooberror)

clf.fit(X, y)
fitted_values = clf.predict(X)
true_values = y
residues = true_values - fitted_values
ooberror = 1 - clf.oob_score_

print ('train_rmse:', np.sqrt(train_mse_list))
print ('test_rmse:', np.sqrt(test_mse_list))
print ('average_train_rmse:', np.sqrt(np.mean(train_mse_list)))
print ('average_test_rmse:', np.sqrt(np.mean(test_mse_list)))
print ('oob_error:', ooberror)

fig = plt.figure()
plt.scatter(range(0, len(y)), true_values, s=0.2)
plt.scatter(range(0, len(y)), fitted_values, s=0.2)
plt.legend(('true', 'fitted'))
plt.xlabel('Data')     
plt.ylabel('Values')
plt.title('Fitted Values VS. True Values For Random Forest Regression')
plt.show()
fig.savefig('2b33')

fig = plt.figure()
plt.scatter(range(0, len(y)), residues, s=0.2)
plt.scatter(range(0, len(y)), fitted_values, s=0.2)
plt.legend(('residues', 'fitted'))
plt.xlabel('Data')     
plt.ylabel('Values')
plt.title('Residues VS. Fitted Values For Random Forest Regression')
plt.show()
fig.savefig('2b34')

clf = RandomForestRegressor(n_estimators=156, max_depth=10,
                                  max_features=3, random_state=2)
clf.fit(X, y)
print(clf.feature_importances_)
features = ['Week #', 'Day of Week', 'Backup Start Time - Hour of Day', 'Work-Flow-ID', 'File Name']
print('The most important feature is:', features[np.argmax(clf.feature_importances_)])


# The following part need to be run on Jupyter Notebook
from sklearn import tree
from IPython.display import Image 
import pydotplus

clf = RandomForestRegressor(n_estimators=156, max_depth=4,
                                  max_features=3, random_state=2)
clf.fit(X, y)
print(clf.feature_importances_)
print('The most important feature is:', features[np.argmax(clf.feature_importances_)])
features = ['Week #', 'Day of Week', 'Backup Start Time - Hour of Day', 'Work-Flow-ID', 'File Name']

dot_data = tree.export_graphviz(clf.estimators_[0], out_file=None, feature_names=features,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data)  
Image(graph.create_png())  