# coding: utf-8
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
from sklearn.model_selection import KFold
import datetime, time
import pytz


pst_tz = pytz.timezone('US/Pacific') 
# pst_tz = pytz.timezone('UTC') 
split1 = datetime.datetime.strptime("01/02/15 08:00", "%d/%m/%y %H:%M")
split1 = pst_tz.localize(split1)
split2 = datetime.datetime.strptime("01/02/15 20:00", "%d/%m/%y %H:%M")
split2 = pst_tz.localize(split2)



path = 'tweet_data/'
path_new = 'test_data/'
hashtags = ['tweets_#gohawks.txt','tweets_#gopatriots.txt','tweets_#nfl.txt','tweets_#patriots.txt',
            'tweets_#sb49.txt', 'tweets_#superbowl.txt']
new_hashtags = ['sample1_period1.txt',
                'sample2_period2.txt',
                'sample3_period3.txt',
                'sample4_period1.txt',
                'sample5_period1.txt',
                'sample6_period2.txt',
                'sample7_period3.txt',
#                 'sample8_period1.txt',
                'sample9_period2.txt',
                'sample10_period3.txt']
# hashtags = ['tweets_#gopatriots.txt']
# new_hashtags = ['sample1_period1.txt']

def rmse(y, pred):
    return sqrt(mean_squared_error(y, pred))

time_post = []
num_retweets = []
num_followers = []
for files in hashtags:
    #load the data
    with open(path+files) as txtfile:
        for row in txtfile:
            data = json.loads(row)
            time_post.append(data['firstpost_date'])
            #num_retweets.append(data['metrics']['citations']['total'])
            #num_followers.append(data['author']['followers'])
    txtfile.close()

total_time = (max(time_post) - min(time_post)) / 3600 #total hours

num_tweets_hour = np.zeros(int(total_time)+1)
num_retweets_hour = np.zeros(int(total_time)+1)
num_followers_hour = np.zeros(int(total_time)+1)
maximum_follower = np.zeros(int(total_time)+1)
favourites_count = np.zeros(int(total_time)+1)
num_ranking_scores = np.zeros(int(total_time)+1)
maximum_retweets = np.zeros(int(total_time)+1)

min_time = min(time_post)
for files in hashtags:
    with open(path+files) as ifile:
        for row in ifile:
            test = json.loads(row)
            index = int((test['firstpost_date']-min_time)/3600)
            num_tweets_hour[index] = num_tweets_hour[index]+1
            num_retweets_hour[index] = num_retweets_hour[index] + test['metrics']['citations']['total']
            num_followers_hour[index] = num_followers_hour[index] + test['author']['followers']
            if( test['author']['followers'] >= maximum_follower[index]):
                maximum_follower[index] = test['author']['followers']
            if( test['metrics']['citations']['total'] >= maximum_retweets[index]):
                maximum_retweets[index] = test['metrics']['citations']['total']
            favourites_count[index] = favourites_count[index] + test['tweet']['user']['favourites_count']
            num_ranking_scores[index] = num_ranking_scores[index] + test['metrics']['ranking_score']
    ifile.close()

#prepocess the data into dataset
features = np.array([num_tweets_hour, num_retweets_hour, num_followers_hour, maximum_follower, 
                 favourites_count, maximum_retweets, num_ranking_scores])
temp_features = features
for i in range(1, 5):
    temp = temp_features[:, i:]
    temp = np.column_stack((temp, np.zeros((7, i))))
    features = np.row_stack((features, temp))
features = features.T
features = features[:-5, :]
outputs = num_tweets_hour[5:]

mae_model1 = []
mae_model2 = []
mae_model3 = []

clf3 = RandomForestRegressor(random_state=0, n_estimators=100, max_depth=10)
clf3.fit(features, outputs)

# predict stage
for files in new_hashtags:
    #load the data
    mae_model3 = []
    time_post = []
    with open(path_new+files) as txtfile:
        for row in txtfile:
            data = json.loads(row)
            time_post.append(data['firstpost_date'])
            #num_retweets.append(data['metrics']['citations']['total'])
            #num_followers.append(data['author']['followers'])
    txtfile.close()

    total_time = (max(time_post) - min(time_post)) / 3600 #total hours
#     print(total_time)

    num_tweets_hour = np.zeros(int(total_time)+1)
    num_retweets_hour = np.zeros(int(total_time)+1)
    num_followers_hour = np.zeros(int(total_time)+1)
    maximum_follower = np.zeros(int(total_time)+1)
    favourites_count = np.zeros(int(total_time)+1)
    num_ranking_scores = np.zeros(int(total_time)+1)
    maximum_retweets = np.zeros(int(total_time)+1)

    min_time = min(time_post)
    with open(path_new+files) as ifile:
        for row in ifile:
            test = json.loads(row)
            index = int((test['firstpost_date']-min_time)/3600)
            num_tweets_hour[index] = num_tweets_hour[index]+1
            num_retweets_hour[index] = num_retweets_hour[index] + test['metrics']['citations']['total']
            num_followers_hour[index] = num_followers_hour[index] + test['author']['followers']
            if( test['author']['followers'] >= maximum_follower[index]):
                maximum_follower[index] = test['author']['followers']
            if( test['metrics']['citations']['total'] >= maximum_retweets[index]):
                maximum_retweets[index] = test['metrics']['citations']['total']
            favourites_count[index] = favourites_count[index] + test['tweet']['user']['favourites_count']
            num_ranking_scores[index] = num_ranking_scores[index] + test['metrics']['ranking_score']
    ifile.close()

    features = np.array([num_tweets_hour, num_retweets_hour, num_followers_hour, maximum_follower, 
                     favourites_count, maximum_retweets, num_ranking_scores])
    temp_features = features
    for i in range(1, 5):
        temp = temp_features[:, i:]
        temp = np.column_stack((temp, np.zeros((7, i))))
        features = np.row_stack((features, temp))
    features = features.T
    features = features[:-5, :]
    outputs = num_tweets_hour[5:]
    predicted_values3 = clf3.predict(features)
    true_values = outputs
    temp = mean_absolute_error(true_values, predicted_values3)
    mae_model3.append(temp)
#     print(true_values)
#     print(predicted_values3)
    print('mae (random forest) for %s is %f' %(files, np.mean(mae_model3)))
    print(num_tweets_hour)
    print(predicted_values3)