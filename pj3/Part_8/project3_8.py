# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 14:24:41 2018

@author: ht
"""

import numpy as np
import matplotlib.pyplot as plt
from surprise import Dataset
from surprise import Reader
from surprise.prediction_algorithms.knns import KNNWithMeans
from surprise.prediction_algorithms.matrix_factorization import NMF, SVD
from surprise.model_selection import split

def search_key(elem):
    return elem[1]

def threshold_rank_filter(test, pred, thre=3, top_t=1):
    data_dict = {}
    pred_dict = {}
    
    for name in test.keys():
        #skip, if |G| = 0 or rating less than t items
        sort_list = sorted(test[name], key=search_key)[::-1]
        if len(sort_list) < top_t:
            continue
        if sort_list[0][1] < thre:
            continue
        #select liked movies from ground truth
        temp = []
        for movie in test[name]:
            if movie[1] < thre:
                temp.append(0)
            else:
                temp.append(1)
        data_dict[name] = temp
        #select top-t movies from prediction
        sort_list = sorted(pred[name], key=search_key)[::-1]
        thre_local = sort_list[top_t-1][1]
        temp = []
        for movie in pred[name]:
            if movie[1] < thre_local:
                temp.append(0)
            else:
                temp.append(1)
        pred_dict[name] = temp
    
    return data_dict, pred_dict
        

def create_dict(data, if_pred = 0):
    dict_data = {}
    idx = if_pred + 2
    for i in range(len(data)):
        if data[i][0] not in dict_data:
            dict_data[data[i][0]] = [(data[i][1], data[i][idx])]
        else:
            dict_data[data[i][0]].append((data[i][1], data[i][idx]))
    return dict_data

def precision_recall(a, b):
    aa = np.asarray(a)
    bb = np.asarray(b)
    if len(a) != len(b):
        raise Exception('Length error!')
    num = np.count_nonzero((aa+bb)/2)
    den_pre = np.count_nonzero(bb)
    den_rec = np.count_nonzero(aa)
    pre = float(num)/float(den_pre)
    rec = float(num)/float(den_rec)
    return pre, rec

#read data
path = '/users/ht/desktop/EE219/proj_3/'
reader = Reader(line_format='user item rating timestamp', sep=',')
data_raw = Dataset.load_from_file(path + 'data/ratings_1.csv', reader=reader)

#define K-fold
num_fold = 10
kf = split.KFold(n_splits = num_fold)

#define model for training
k_min = 24
sim_options = {'name': 'pearson', 'user_based': True}
knn = KNNWithMeans(k=k_min, sim_options=sim_options)

#train, test and rank
top_t_list = range(1,26)
pre_list_knn = []
rec_list_knn = []
for top_t in top_t_list:
    pre = 0
    rec = 0
    for trainset, testset in kf.split(data_raw):
        knn.fit(trainset)
        prediction = knn.test(testset)
        G = create_dict(testset)
        G_s = create_dict(prediction, if_pred=1)
        R, R_s = threshold_rank_filter(G, G_s, thre=3, top_t=top_t)
        #precision and recall for each fold
        pre_fold = 0
        rec_fold = 0
        for key in R.keys():
            pre_temp, rec_temp = precision_recall(R[key], R_s[key])
            pre_fold += pre_temp
            rec_fold += rec_temp
        pre += pre_fold/len(R)
        rec += rec_fold/len(R)
     
    pre_list_knn.append(pre/num_fold)
    rec_list_knn.append(rec/num_fold)

fig1 = plt.figure()
plt.plot(top_t_list, pre_list_knn)
plt.xlabel('t')     
plt.ylabel('Average Precision')
plt.title('Average Precision VS T')
plt.grid(color='0.7', linestyle='-', linewidth=1)
plt.xlim([0.0, 25.0])
plt.ylim([0.0, 1.0])
plt.show()
fig1.savefig(path+'fig/Part_8_knn_pre.png')

fig2 = plt.figure()
plt.plot(top_t_list, rec_list_knn)
plt.xlabel('t')     
plt.ylabel('Average Recall')
plt.title('Average Recall VS T')
plt.grid(color='0.7', linestyle='-', linewidth=1)
plt.xlim([0.0, 25.0])
plt.ylim([0.0, 1.0])
plt.show()
fig2.savefig(path+'fig/Part_8_knn_rec.png')

fig3 = plt.figure()
plt.plot(rec_list_knn, pre_list_knn)
plt.xlabel('Average Recall')     
plt.ylabel('Average Precision')
plt.title('Average Precision VS Average Recall')
plt.grid(color='0.7', linestyle='-', linewidth=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.show()
fig3.savefig(path+'fig/Part_8_knn_preVSrec.png')



#define model for training
k_min_rmse = 18
nnmf = NMF(n_factors = k_min_rmse, random_state = 1)

#train, test and rank
top_t_list = range(1,26)
pre_list_nnmf = []
rec_list_nnmf = []
for top_t in top_t_list:
    pre = 0
    rec = 0
    for trainset, testset in kf.split(data_raw):
        nnmf.fit(trainset)
        prediction = nnmf.test(testset)
        G = create_dict(testset)
        G_s = create_dict(prediction, if_pred=1)
        R, R_s = threshold_rank_filter(G, G_s, thre=3, top_t=top_t)
        #precision and recall for each fold
        pre_fold = 0
        rec_fold = 0
        for key in R.keys():
            pre_temp, rec_temp = precision_recall(R[key], R_s[key])
            pre_fold += pre_temp
            rec_fold += rec_temp
        pre += pre_fold/len(R)
        rec += rec_fold/len(R)
     
    pre_list_nnmf.append(pre/num_fold)
    rec_list_nnmf.append(rec/num_fold)

fig11 = plt.figure()
plt.plot(top_t_list, pre_list_nnmf)
plt.xlabel('t')     
plt.ylabel('Average Precision')
plt.title('Average Precision VS T')
plt.grid(color='0.7', linestyle='-', linewidth=1)
plt.xlim([0.0, 25.0])
plt.ylim([0.0, 1.0])
plt.show()
fig11.savefig(path+'fig/Part_8_nnmf_pre.png')

fig22 = plt.figure()
plt.plot(top_t_list, rec_list_nnmf)
plt.xlabel('t')     
plt.ylabel('Average Recall')
plt.title('Average Recall VS T')
plt.grid(color='0.7', linestyle='-', linewidth=1)
plt.xlim([0.0, 25.0])
plt.ylim([0.0, 1.0])
plt.show()
fig22.savefig(path+'fig/Part_8_nnmf_rec.png')

fig33 = plt.figure()
plt.plot(rec_list_nnmf, pre_list_nnmf)
plt.xlabel('Average Recall')     
plt.ylabel('Average Precision')
plt.title('Average Precision VS Average Recall')
plt.grid(color='0.7', linestyle='-', linewidth=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.show()
fig33.savefig(path+'fig/Part_8_nnmf_preVSrec.png')



#define model for training
k_min_rmse = 16
mf = SVD(n_factors = k_min_rmse, random_state = 1, biased=True)

#train, test and rank
top_t_list = range(1,26)
pre_list_mf = []
rec_list_mf = []
for top_t in top_t_list:
    pre = 0
    rec = 0
    for trainset, testset in kf.split(data_raw):
        mf.fit(trainset)
        prediction = mf.test(testset)
        G = create_dict(testset)
        G_s = create_dict(prediction, if_pred=1)
        R, R_s = threshold_rank_filter(G, G_s, thre=3, top_t=top_t)
        #precision and recall for each fold
        pre_fold = 0
        rec_fold = 0
        for key in R.keys():
            pre_temp, rec_temp = precision_recall(R[key], R_s[key])
            pre_fold += pre_temp
            rec_fold += rec_temp
        pre += pre_fold/len(R)
        rec += rec_fold/len(R)
     
    pre_list_mf.append(pre/num_fold)
    rec_list_mf.append(rec/num_fold)

fig111 = plt.figure()
plt.plot(top_t_list, pre_list_mf)
plt.xlabel('t')     
plt.ylabel('Average Precision')
plt.title('Average Precision VS T')
plt.grid(color='0.7', linestyle='-', linewidth=1)
plt.xlim([0.0, 25.0])
plt.ylim([0.0, 1.0])
plt.show()
fig111.savefig(path+'fig/Part_8_mf_pre.png')

fig222 = plt.figure()
plt.plot(top_t_list, rec_list_mf)
plt.xlabel('t')     
plt.ylabel('Average Recall')
plt.title('Average Recall VS T')
plt.grid(color='0.7', linestyle='-', linewidth=1)
plt.xlim([0.0, 25.0])
plt.ylim([0.0, 1.0])
plt.show()
fig222.savefig(path+'fig/Part_8_mf_rec.png')

fig333 = plt.figure()
plt.plot(rec_list_mf, pre_list_mf)
plt.xlabel('Average Recall')     
plt.ylabel('Average Precision')
plt.title('Average Precision VS Average Recall')
plt.grid(color='0.7', linestyle='-', linewidth=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.show()
fig333.savefig(path+'fig/Part_8_mf_preVSrec.png')


##plot overall
fig = plt.figure()
lw = 1
plt.plot(rec_list_mf, pre_list_mf, color='aqua',
         lw=lw, label='MF with biase')
plt.plot(rec_list_nnmf, pre_list_nnmf, color='cornflowerblue',
         lw=lw, label='NNMF')
plt.plot(rec_list_knn, pre_list_knn, color='darkorange',
         lw=lw, label='K-NN')
plt.xlabel('Average Recall')     
plt.ylabel('Average Precision')
plt.title('Average Precision VS Average Recall')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.legend(loc="lower right")
plt.show()
fig.savefig(path+'fig/Part_8_overall_preVSrec.png')