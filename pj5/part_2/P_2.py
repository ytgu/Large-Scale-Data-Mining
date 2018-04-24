"""
Build 4 different models with 2 different dimension reduction methods.
models: SVM, Naive Bayesian, Logistic Regression and Random Forest
dimension reduction methods: LSI(SVD), NMF

author: Haitao Wang
Date: 3/8/2018
"""

import json
import os
import re
import time
import scipy as sc
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import roc_curve as rc
from sklearn.metrics import confusion_matrix as cm
import itertools
import sklearn.metrics as m
from sklearn.model_selection import KFold
from sklearn.decomposition import NMF
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

####### plot function defined ########
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
####################################


#find city names
path = os.path.dirname(os.path.realpath(__file__))
with open(path+'/data/stats.txt') as f:
    for line in f:
        data = json.loads(line)

regex_wa = ['[^a-zA-Z]wa[^a-zA-Z]', '[^a-zA-Z]wa$', 'washington', '^wa[^a-zA-Z]', '^wa$']
regex_ma = ['[^a-zA-Z]ma[^a-zA-Z]', '[^a-zA-Z]ma$', 'massachusetts', '^ma[^a-zA-Z]', '^ma$']

loc_wa_list = []
for key in data.keys():
    for reg in regex_wa:
        match = re.search(reg, key.lower())
        if reg == 'washington':
            try:
                match_r = next(r for r in ['dc', 'd.c'] if re.search(r, key.lower()) is not None)
            except:
                match_r = None
        if match and not match_r:
            loc_wa_list.append(key)

loc_ma_list = []
for key in data.keys():
    for reg in regex_ma:
        match = re.search(reg, key.lower())
        if match:
            loc_ma_list.append(key)

reg_wa = ['^(.*), wa$', '^(.*),wa$', '^(.*), wa[.*]', '^(.*),wa[^a-z]']
reg_ma = ['^(.*), ma$', '^(.*),ma$', '^(.*), ma[.*]', '^(.*),ma[^a-z]']

print len(loc_wa_list)
print len(loc_ma_list)

start_time = time.time()
loc_wa = {}
loc_ma = {}
for key in data.keys():
    if_wa = 0
    for reg in reg_wa:
        match = re.search(reg, key.lower())
        if match:
            if match.groups()[0].lower() not in loc_wa:
                loc_wa[match.groups()[0].lower()] = data[key]
            else:
                loc_wa[match.groups()[0].lower()] += data[key]
            if_wa = 1
            break
    if not if_wa:
        for reg in reg_ma:
            match = re.search(reg, key.lower())
            if match:
                if match.groups()[0].lower() not in loc_ma:
                    loc_ma[match.groups()[0].lower()] = data[key]
                else:
                    loc_ma[match.groups()[0].lower()] += data[key]
                break

elapsed_time = time.time() - start_time
print elapsed_time

city_wa = []
for city, num in sorted(loc_wa.items(), key=lambda x:x[1])[::-1]:
    if num >= 50:
        city_wa.append(city.encode('ascii','ignore'))

city_ma = []
for city, num in sorted(loc_ma.items(), key=lambda x:x[1])[::-1]:
    if num >= 50:
        city_ma.append(city.encode('ascii','ignore'))
print city_wa
print city_ma

#extract useful data
regex_wa = ['[^a-zA-Z]wa[^a-zA-Z]', '[^a-zA-Z]wa$', 'washington$', '^wa[^a-zA-Z]', '^wa$', 'washington state']
regex_ma = ['[^a-zA-Z]ma[^a-zA-Z]', '[^a-zA-Z]ma$', 'massachusetts$', '^ma[^a-zA-Z]', '^ma$']
regex_wa += city_wa
regex_ma += city_ma

data = []
label = []
start_time = time.time()
with open(path+"/data/tweets_#superbowl.txt", "r") as f:
    line = f.readlines()
for l in line:
    text = json.loads(l)
    if_append = 0
    for reg in regex_wa:
        match = re.search(reg, text['tweet']['user']['location'].lower())
        if reg == 'washington$':
            try:
                match_r = next(r for r in ['dc', 'd.c'] if re.search(r, text['tweet']['user']['location'].lower()) is not None)
            except:
                match_r = None
        if match and not match_r:
            data.append(text['tweet']['text'])
            label.append(0)
            if_append = 1
            break
    if not if_append:
        for reg in regex_ma:
            match = re.search(reg, text['tweet']['user']['location'].lower())
            if match:
                data.append(text['tweet']['text'])
                label.append(1)
                break
elapsed_time = time.time() - start_time
print elapsed_time
print len(data)
print len(label)

num_ma = sum(label)
num_wa = len(label) - num_ma
print 'Number of Massachusetts:'
print num_ma
print 'Number of Washington:'
print num_wa

#preprocess data
ps = PorterStemmer()

class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        return lambda X: (ps.stem(doc) for doc in analyzer(X))

vectorizer = StemmedTfidfVectorizer(min_df=5, stop_words='english')
X = vectorizer.fit_transform(data)
num_samples, num_features = X.shape
print("#samples: %d, #features: %d" % (num_samples, num_features))

#SVD
u,s,v = sc.sparse.linalg.svds(X.T, k=50)
features = u.T * X.T
features = features.T
"""
#NMF
nmf = NMF(n_components=50, init='random', random_state=42)
features = nmf.fit_transform(X)
"""

#SVM
# define model
model_hsv = SVC(C=1000)
model_ssv = SVC(C=1)

# preprocess data
label = np.asarray(label)
print type(features)
print features.shape
print type(label)
print len(label)

# train model
acc_h = []
acc_s = []

kf = KFold(n_splits=10, shuffle=True, random_state=42)
for train_index, test_index in kf.split(features):
    #print type(train_index)
    #print len(test_index)
    start_time = time.time()
    model_hsv.fit(features[train_index], label[train_index])
    model_ssv.fit(features[train_index], label[train_index])

    # test model
    features_t = features[test_index]
    test_op = label[test_index]
    #true_op.append(test_op)
    pred_hsv = model_hsv.predict(features_t)
    score_hsv = model_hsv.score(features_t, test_op)
    pred_ssv = model_ssv.predict(features_t)
    score_ssv = model_ssv.score(features_t, test_op)
    acc_h.append(score_hsv)
    acc_s.append(score_ssv)
    print 'Time spent in each fold:'
    print time.time() - start_time

# plot ROC
y_score_hsv = model_hsv.decision_function(features_t)
fpr_h, tpr_h, _ = rc(test_op, y_score_hsv)
y_score_ssv = model_ssv.decision_function(features_t)
fpr_s, tpr_s, _ = rc(test_op, y_score_ssv)
fig1 = plt.figure()
lw = 1
plt.plot(fpr_h, tpr_h, color='darkorange',
         lw=lw, label='ROC curve (Soft SVM)')
plt.plot(fpr_s, tpr_s, color='deeppink', lw=lw,
         label='ROC curve (Hard SVM)')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
fig1.savefig(path+'/fig/SVM_ROC.png')

# plot confusion matrix
C_h = cm(test_op, pred_hsv)
C_s = cm(test_op, pred_ssv)

fig2 = plt.figure()
class_names = ['Washington', 'Massachusetts']
plot_confusion_matrix(C_h, classes=class_names,
                      title='Confusion matrix, without normalization(Soft)')
fig2.savefig(path+'/fig/SVM_CM_NN_S.png')
fig3 = plt.figure()
plot_confusion_matrix(C_h, classes=class_names, normalize=True,
                      title='Normalized confusion matrix(Soft)')
fig3.savefig(path+'/fig/SVM_CM_ND_S.png')
fig4 = plt.figure()
plot_confusion_matrix(C_s, classes=class_names,
                      title='Confusion matrix, without normalization(Hard)')
fig4.savefig(path+'/fig/SVM_CM_NN_H.png')
fig5 = plt.figure()
plot_confusion_matrix(C_s, classes=class_names, normalize=True,
                      title='Normalized confusion matrix(Hard)')
fig5.savefig(path+'/fig/SVM_CM_ND_H.png')


#naive bayes
model_mnb = MultinomialNB()
model_gnb = GaussianNB()

#preprocess features for mnb
scaler = MinMaxScaler()
features_mod = scaler.fit_transform(features)
#features_mod = features
score_list_mnb = []
score_list_gnb = []
for train_index, test_index in kf.split(features):
    start_time = time.time()
    #train
    model_mnb.fit(features_mod[train_index], label[train_index])
    model_gnb.fit(features[train_index], label[train_index])
    #test
    score_list_mnb.append(model_mnb.score(features_mod[test_index], label[test_index]))
    pred_prob_mnb = model_mnb.predict_proba(features_mod[test_index])
    pred_mnb = model_mnb.predict(features_mod[test_index])
    score_list_gnb.append(model_gnb.score(features[test_index], label[test_index]))
    pred_prob_gnb = model_gnb.predict_proba(features[test_index])
    pred_gnb = model_gnb.predict(features[test_index])
    y_score_mnb = pred_prob_mnb[:, 1] - pred_prob_mnb[:, 0]
    y_score_gnb = pred_prob_gnb[:, 1] - pred_prob_gnb[:, 0]
    print 'Time spent in each fold:'
    print time.time() - start_time

# plot ROC
fpr_m, tpr_m, _ = rc(test_op, y_score_mnb)
fpr_g, tpr_g, _ = rc(test_op, y_score_gnb)
fig11 = plt.figure()
lw = 1
plt.plot(fpr_m, tpr_m, color='darkorange',
         lw=lw, label='ROC curve (MNB)')
plt.plot(fpr_g, tpr_g, color='deeppink', lw=lw,
         label='ROC curve (GNB)')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
fig11.savefig(path+'/fig/NB_ROC.png')

# plot confusion matrix
C_h = cm(test_op, pred_hsv)
C_s = cm(test_op, pred_ssv)

fig12 = plt.figure()
class_names = ['Washington', 'Massachusetts']
plot_confusion_matrix(C_h, classes=class_names,
                      title='Confusion matrix, without normalization(Multinomial)')
fig12.savefig(path+'/fig/NB_CM_NN_M.png')
fig13 = plt.figure()
plot_confusion_matrix(C_h, classes=class_names, normalize=True,
                      title='Normalized confusion matrix(Multinomial)')
fig13.savefig(path+'/fig/NB_CM_ND_M.png')
fig14 = plt.figure()
plot_confusion_matrix(C_s, classes=class_names,
                      title='Confusion matrix, without normalization(Gaussian)')
fig14.savefig(path+'/fig/NB_CM_NN_G.png')
fig15 = plt.figure()
plot_confusion_matrix(C_s, classes=class_names, normalize=True,
                      title='Normalized confusion matrix(Gaussian)')
fig15.savefig(path+'/fig/NB_CM_ND_G.png')


#logistic regression
model_lr = LogisticRegression(C=1000)
score_list_lr = []
for train_index, test_index in kf.split(features):
    start_time = time.time()
    #train
    model_lr.fit(features[train_index], label[train_index])
    #test
    score_list_lr.append(model_lr.score(features[test_index], label[test_index]))
    pred_lr = model_lr.predict(features[test_index])
    print 'Time spent in each fold:'
    print time.time() - start_time

#plot ROC
y_score_lr = model_lr.decision_function(features[test_index])
fpr_l,tpr_l,_ = rc(label[test_index], y_score_lr)
fig_21 = plt.figure()
lw = 1
plt.plot(fpr_l, tpr_l, color='black', linestyle='-.',
         lw=lw, label='ROC curve (LogisticR)')
plt.plot(fpr_m, tpr_m, color='aqua', linestyle=':',
         lw=lw, label='ROC curve (Multinomial)')
plt.plot(fpr_g, tpr_g, color='cornflowerblue', lw=lw,
         label='ROC curve (Gaussian)', linestyle=':')
plt.plot(fpr_h, tpr_h, color='darkorange',
         lw=lw, label='ROC curve (Soft SVM)')
plt.plot(fpr_s, tpr_s, color='deeppink', lw=lw,
         label='ROC curve (Hard SVM)')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
fig_21.savefig(path+'/fig/LR_ROC.png')

#plot confusion matrix
C_l = cm(label[test_index], pred_lr)
fig_22 = plt.figure()
class_names = ['Washington', 'Massachusetts']
plot_confusion_matrix(C_l, classes=class_names,
                      title='Confusion matrix, without normalization(LR)')
fig_22.savefig(path+'/fig/LR_CM_NN.png')
fig_23 = plt.figure()
plot_confusion_matrix(C_l, classes=class_names, normalize=True,
                      title='Normalized confusion matrix(LR)')
fig_23.savefig(path+'/fig/LR_CM_ND.png')



#Random forest
model_rf = RandomForestClassifier(n_estimators=20, max_features=5, bootstrap=True, random_state=42)
score_list_rf = []
for train_index, test_index in kf.split(features):
    start_time = time.time()
    #train
    model_rf.fit(features[train_index], label[train_index])
    #test
    score_list_rf.append(model_rf.score(features[test_index], label[test_index]))
    pred_rf = model_rf.predict(features[test_index])
    pred_prob_rf = model_rf.predict_proba(features[test_index])
    y_score_rf = pred_prob_rf[:, 1] - pred_prob_rf[:, 0]
    print 'Time spent in each fold:'
    print time.time() - start_time

#plot roc
fpr_r,tpr_r,_ = rc(label[test_index], y_score_rf)
fig_31 = plt.figure()
lw = 1
plt.plot(fpr_r, tpr_r, color='red', linestyle='-.',
         lw=lw, label='ROC curve (Random Forest)')
plt.plot(fpr_l, tpr_l, color='black', linestyle='-.',
         lw=lw, label='ROC curve (LogisticR)')
plt.plot(fpr_m, tpr_m, color='aqua', linestyle=':',
         lw=lw, label='ROC curve (Multinomial)')
plt.plot(fpr_g, tpr_g, color='cornflowerblue', lw=lw,
         label='ROC curve (Gaussian)', linestyle=':')
plt.plot(fpr_h, tpr_h, color='darkorange',
         lw=lw, label='ROC curve (Soft SVM)')
plt.plot(fpr_s, tpr_s, color='deeppink', lw=lw,
         label='ROC curve (Hard SVM)')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
fig_31.savefig(path+'/fig/RF_ROC.png')

#plot confusion matrix
C_r = cm(label[test_index], pred_rf)
fig_32 = plt.figure()
class_names = ['Washington', 'Massachusetts']
plot_confusion_matrix(C_r, classes=class_names,
                      title='Confusion matrix, without normalization(RF)')
fig_32.savefig(path+'/fig/RF_CM_NN.png')
fig_33 = plt.figure()
plot_confusion_matrix(C_r, classes=class_names, normalize=True,
                      title='Normalized confusion matrix(RF)')
fig_33.savefig(path+'/fig/RF_CM_ND.png')


#write stats into output file
with open(path+'/data/result_NMF.txt', 'w') as f:
    f.write('score of hsv:')
    f.write(str(acc_h)+'\n')
    f.write(str(m.classification_report(test_op, pred_hsv, target_names=['Washington', 'Massachusetts'])))
    f.write('score of ssv:')
    f.write(str(acc_s)+'\n')
    f.write(str(m.classification_report(test_op, pred_ssv, target_names=['Washington', 'Massachusetts'])))
    f.write('score of mnb:')
    f.write(str(score_list_mnb)+'\n')
    f.write(str(m.classification_report(label[test_index], pred_mnb, target_names=['Washington', 'Massachusetts'])))
    f.write('score of gnb:')
    f.write(str(score_list_gnb)+'\n')
    f.write(str(m.classification_report(label[test_index], pred_gnb, target_names=['Washington', 'Massachusetts'])))
    f.write('score of lr:')
    f.write(str(score_list_lr)+'\n')
    f.write(str(m.classification_report(label[test_index], pred_lr, target_names=['Washington', 'Massachusetts'])))
    f.write('score of rf:')
    f.write(str(score_list_rf)+'\n')
    f.write(str(m.classification_report(label[test_index], pred_rf, target_names=['Washington', 'Massachusetts'])))
