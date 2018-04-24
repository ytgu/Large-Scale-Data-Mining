#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 20:08:41 2018

@author: AndrewXu
"""
from datetime import datetime
from textblob import TextBlob
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
import json
import pandas as pd
import re
import matplotlib.pyplot as plt

num_of_lines = [188136, 26232]
num_fold = 10
max_neurons = 100
frequency = '10Min'
#start_interval = datetime(2015,1,14,6,0,0)
#end_interval = datetime(2015,2,2,20,0,0)
start_interval = datetime(2015,2,1,7,0,0)
end_interval = datetime(2015,2,1,23,0,0)

file = open("log.txt", "w")
path = ['/Users/Weqian/Downloads/tweet_data/tweets_#gohawks.txt', \
        '/Users/Weqian/Downloads/tweet_data/tweets_#gopatriots.txt']

#reference: https://www.geeksforgeeks.org/twitter-sentiment-analysis-using-python/
def clean_tweet(tweet):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

def get_tweet_sentiment(tweet):
    
    sentiment_analysis = TextBlob(clean_tweet(tweet))
    return sentiment_analysis.polarity

def rmse(y, pred):
    return sqrt(mean_squared_error(y, pred))

def generate_nn_input(data):
    temp_hours = data.groupby(pd.TimeGrouper(freq = frequency))

    X = np.zeros((len(temp_hours), 11))
    Y = np.zeros((len(temp_hours), 1))
    
    pos_senti_ratio = np.zeros((len(temp_hours),1))
    
    for i,(interval, group) in enumerate(temp_hours):
        X[i, 0] = group.tweet_count.sum()      #total count of tweets
        X[i, 1] = group.retweet_count.sum()    #total count of retweets
        X[i, 2] = group.followers.sum()        #total number of followers
        X[i, 3] = group.followers.max()        #maximum of followers
        X[i, 4] = group.no_hashtag.sum()       #total count of hashtags
        X[i, 5] = interval.hour                #hour of the day
        X[i, 6] = group.impression_count.sum() #Sum of impression count
        X[i, 7] = group.favorite_count.sum()   #Sum of favorite count
        X[i, 8] = group.ranking_score.sum()    #Sum of rankings
        
        tweet_num = group.tweet_count.sum()
        num_pos_senti = sum(j > 0 for j in group.sentiment)
        if (tweet_num == 0):
            pos_senti_ratio[i] = 0;
        else:
            pos_senti_ratio[i] = num_pos_senti/tweet_num
    
    for i in range(2, len(temp_hours)):
        X[i, 9] = pos_senti_ratio[i]
        X[i, 10] = pos_senti_ratio[i-1]
    
    X = X[2:,:]
    Y = pos_senti_ratio[2:,:]
    
    X = np.nan_to_num(X[:-1])
    Y = Y[1:]
    
    return X, Y

def train_and_test(X, Y):
    
    total_train_rmse = []
    total_test_rmse = []
    kf = KFold(n_splits = num_fold)
    
    for neuron_no in range(1, max_neurons+1, 2):
        print ('training.....neurons=%d' % neuron_no)
        relu_nn = MLPRegressor(hidden_layer_sizes = (neuron_no,), \
                               activation = 'relu', verbose = False)
        train_rmse_list = []
        test_rmse_list = []
        for train_index, test_index in kf.split(X):
            relu_nn.fit(X[train_index], Y[train_index].ravel())
            train_rmse = rmse(Y[train_index].ravel(),relu_nn.predict(X[train_index]))
            test_rmse = rmse(Y[test_index].ravel(),relu_nn.predict(X[test_index]))
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
    plt.plot(range(1, max_neurons+1, 2), total_test_rmse, 'b')
    plt.xlabel('Number of hidden neurons (Activity Function = Relu)')
    plt.ylabel('Test RMSE')
    plt.show()
    fig.savefig('Test_rmse.png')
    
    total_train_rmse = np.asarray(total_train_rmse)
    total_test_rmse = np.asarray(total_test_rmse)
    
    best_neuron_no = (np.where(total_train_rmse == total_train_rmse.min())[0]) * 2 + 1
    file.write("Best number of neuron is %d\n" % best_neuron_no)
    file.write("The RMSE value related is %s\n" % total_train_rmse.min())
    print ("Best number of neuron is %d" % best_neuron_no)
    print ("The RMSE value related is %s" % total_train_rmse.min())
    
    relu_nn = MLPRegressor(hidden_layer_sizes = (best_neuron_no[0],), \
                            activation = 'relu', verbose = False)
    
    train_rmse_list = []
    test_rmse_list = []
    for train_index, test_index in kf.split(X):
        relu_nn.fit(X[train_index], Y[train_index].ravel())
        train_rmse = rmse(Y[train_index].ravel(),relu_nn.predict(X[train_index]))
        test_rmse = rmse(Y[test_index].ravel(),relu_nn.predict(X[test_index]))
        train_rmse_list.append(train_rmse)
        test_rmse_list.append(test_rmse)

    fitted_values = relu_nn.predict(X)
    true_values = Y
    residues = fitted_values - true_values

    fig = plt.figure()
    plt.plot(true_values, fitted_values,'ro',linewidth=0.1, markersize = 1)
    plt.plot([0,1],[0,1], '--', color='blue')
    plt.xlabel('True values')     
    plt.ylabel('Fitted values')
    plt.title('Fitted Values VS. True Values for Neural Network (Relu)')
    plt.show()
    fig.savefig('Relu_fitted.png')

    fig = plt.figure()
    plt.plot(true_values, residues,'ro',linewidth=0.1, markersize = 1)
    plt.plot([0,1],[0,0], '--', color='blue')
    plt.xlabel('True values')     
    plt.ylabel('Residues')
    plt.title('Residues VS. True Values for Neural Network (Relu)')
    plt.show()
    fig.savefig('Relu_res.png')

avg_pos_hawks = []
avg_neg_hawks = []
pos_ratio_hawks = []
neg_ratio_hawks = []

avg_pos_patriots = []
avg_neg_patriots = []
pos_ratio_patriots = []
neg_ratio_patriots = []



datafile_no = 0;
for i in path:
    count = 0;
    with open(i,'r') as myfile:
        all_data = pd.DataFrame(index=range(num_of_lines[datafile_no]), columns = ['time','content','tweet_count', \
                                'retweet_count', 'followers', 'sentiment', 'no_hashtag', \
                                'impression_count', 'favorite_count', 'ranking_score'])
        i = 0
        for line in myfile.readlines():
            tweet = json.loads(line)
            all_data.set_value(i, 'content', tweet['title'])
            time = datetime.fromtimestamp(tweet['firstpost_date'])
            all_data.set_value(i, 'time', time)
            all_data.set_value(i, 'tweet_count', 1)
            all_data.set_value(i,'retweet_count', tweet['metrics']['citations']['total'])
            all_data.set_value(i, 'followers', tweet['author']['followers'])
            all_data.set_value(i, 'no_hashtag', tweet['title'].count('#'))
            all_data.set_value(i, 'sentiment', get_tweet_sentiment(tweet['title']))
            all_data.set_value(i, 'impression_count', tweet.get('metrics').get('impressions'))
            all_data.set_value(i, 'favorite_count', tweet.get('tweet').get('favorite_count'))
            all_data.set_value(i, 'ranking_score', tweet.get('metrics').get('ranking_score'))
            i = i + 1

        
        all_data = all_data[(all_data.time > start_interval) & (all_data.time < end_interval)]
    
        all_data = all_data.set_index('time')
        
        temp_content = all_data['content'].values.tolist()
        temp_sentiment = all_data['sentiment'].values.tolist()
        np_sentiment = np.asarray(temp_sentiment)
        
        
        index_top = np.asarray(np.where(np_sentiment == 1))
        count = 1
        for i in index_top[0]:
            print('%d.\t%s' % (count, temp_content[i]))
            file.write('%d.\t%s\n' % (count, temp_content[i]))
            count += 1
        
        print('\n\n\n')
        file.write('\n\n\n')
        
        index_low = np.asarray(np.where(np_sentiment == -1))
        count = 1
        for i in index_low[0]:
            print('%d.\t%s' % (count, temp_content[i]))
            file.write('%d.\t%s\n' % (count, temp_content[i]))
            count += 1
        
        
        hours = all_data.groupby(pd.TimeGrouper(freq = frequency))
        
        num_tweet_counter = []
        num_pos_senti_counter = []
        num_neutral_senti_counter = []
        num_neg_senti_counter = []
        pos_vs_total_ratio = []
        neutral_vs_total_ratio = []
        neg_vs_total_ratio = []
        avg_pos_senti = []
        avg_neg_senti = []
        avg_total_senti = []
        
        for i, (interval, group) in enumerate(hours):
            num_tweets = group.tweet_count.sum()
            num_tweet_counter.append(num_tweets)
            num_pos_senti_counter.append(sum(j > 0 for j in group.sentiment))
            num_neutral_senti_counter.append(sum(j == 0 for j in group.sentiment))
            num_neg_senti_counter.append(sum(j < 0 for j in group.sentiment))
            
            
            if (num_tweets != 0):
                avg_pos_senti.append(sum(j for j in group.sentiment if j > 0) / num_tweets)
                avg_neg_senti.append(sum(j for j in group.sentiment if j < 0) / num_tweets)
                avg_total_senti.append((sum(j for j in group.sentiment if j > 0) + \
                                        -1 * sum(j for j in group.sentiment if j < 0)) / num_tweets)
                pos_vs_total_ratio.append(num_pos_senti_counter[i]/num_tweets)
                neutral_vs_total_ratio.append(num_neutral_senti_counter[i]/num_tweets)
                neg_vs_total_ratio.append(num_neg_senti_counter[i]/num_tweets)
                
            else:
                avg_pos_senti.append(0)
                avg_neg_senti.append(0)
                avg_total_senti.append(0)
                pos_vs_total_ratio.append(0.0)
                neutral_vs_total_ratio.append(0.0)
                neg_vs_total_ratio.append(0.0)
                
        avg_neg_senti = np.absolute(avg_neg_senti)
                
        if (datafile_no == 0):
            avg_pos_hawks = avg_pos_senti
            avg_neg_hawks = avg_neg_senti
            pos_ratio_hawks = pos_vs_total_ratio
            neg_ratio_hawks = neg_vs_total_ratio
        else:
            avg_pos_patriots = avg_pos_senti
            avg_neg_patriots = avg_neg_senti
            pos_ratio_patriots = pos_vs_total_ratio
            neg_ratio_patriots = neg_vs_total_ratio
            
            
        fig = plt.figure()
        plt.bar(range(len(num_tweet_counter)), num_tweet_counter, width = 1.2)
        plt.xlabel('Times %s' % frequency)
        plt.ylabel('Number of tweets')
        plt.show()
        fig.savefig('num_tweet_vs_time_%d.png' % datafile_no)
        
        
        fig = plt.figure()
        plt.plot(range(len(avg_pos_senti)), avg_pos_senti, 'b', linewidth = 0.5, label = 'positive')
        plt.plot(range(len(avg_neg_senti)), avg_neg_senti, 'r', linewidth = 0.5, label = 'negative')
        plt.legend()
        plt.xlabel('Times %s' % frequency)
        plt.ylabel('Average polarity of tweets')
        plt.show()
        fig.savefig('avg_posneg_polarity_vs_time_%d.png' % datafile_no)
        
        fig = plt.figure()
        plt.plot(range(len(pos_vs_total_ratio)),pos_vs_total_ratio,'b',linewidth = 0.5, label = 'positive')
        plt.plot(range(len(neg_vs_total_ratio)),neg_vs_total_ratio,'r',linewidth = 0.5, label = 'negative')
        plt.legend()
        plt.xlabel('Times %s' % frequency)
        plt.ylabel('Ratio')
        plt.title('Number of positive tweets vs. total tweets')
        plt.show()
        fig.savefig('posneg_tweet_vs_total_%d.png' % datafile_no)
        
        #x, y = generate_nn_input(all_data)
        #train_and_test(x, y)
        
    datafile_no += 1  

fig = plt.figure()
plt.plot(range(len(avg_pos_hawks)), avg_pos_hawks, 'b', linewidth = 0.5, label = 'hawks positive')
plt.plot(range(len(avg_neg_hawks)), avg_neg_hawks, 'r', linewidth = 0.5, label = 'hawks negative')
plt.plot(range(len(avg_pos_patriots)), avg_pos_patriots, 'g', linewidth = 0.5, label = 'patriots positive')
plt.plot(range(len(avg_neg_patriots)), avg_neg_patriots, 'c', linewidth = 0.5, label = 'patriots negative')
plt.legend()
plt.xlabel('Times %s' % frequency)
plt.ylabel('Average polarity of tweets')
plt.show()
fig.savefig('avg_posneg_polarity_vs_time_%d.png' % datafile_no)
        
fig = plt.figure()
plt.plot(range(len(pos_ratio_hawks)),pos_ratio_hawks,'b',linewidth = 0.5, label = 'hawks positive')
plt.plot(range(len(neg_ratio_hawks)),neg_ratio_hawks,'r',linewidth = 0.5, label = 'hawks negative')
plt.plot(range(len(pos_ratio_patriots)),pos_ratio_patriots,'g',linewidth = 0.5, label = 'patriots positive')
plt.plot(range(len(neg_ratio_patriots)),neg_ratio_patriots,'c',linewidth = 0.5, label = 'patriots negative')
plt.legend()
plt.xlabel('Times %s' % frequency)
plt.ylabel('Ratio')
plt.title('Number of positive tweets vs. total tweets')
plt.show()
fig.savefig('posneg_tweet_vs_total_%d.png' % datafile_no)
    
file.close()
    

        
        
    
        