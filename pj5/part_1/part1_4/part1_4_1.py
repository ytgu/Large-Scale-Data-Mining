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
hashtags = ['tweets_#gohawks.txt','tweets_#gopatriots.txt','tweets_#nfl.txt','tweets_#patriots.txt',
            'tweets_#sb49.txt', 'tweets_#superbowl.txt']
# hashtags = ['tweets_#gohawks.txt']

def rmse(y, pred):
    return sqrt(mean_squared_error(y, pred))

for files in hashtags:
    time_post = [[], [], []]
    total_time = [0, 0, 0]
    num_retweets = []
    num_followers = []
    
    #load the data
    with open(path+files) as txtfile:
        for row in txtfile:
            data = json.loads(row)
            temp = datetime.datetime.fromtimestamp(data['citation_date'], pst_tz)
            if(temp < split1):
                time_post[0].append(data['citation_date'])
            elif((temp>=split1) & (temp<=split2)):
                time_post[1].append(data['citation_date'])
            else:
                time_post[2].append(data['citation_date'])
            #num_retweets.append(data['metrics']['citations']['total'])
            #num_followers.append(data['author']['followers'])
    txtfile.close()
    #calculate the output
    print('=========================================================================')
    print('hashtage is: %s \n' %files)
    
    model_str = ['Before Feb. 1, 8:00 a.m.', 
                 'Between Feb. 1, 8:00 a.m. and 8:00 p.m.',
                 'After Feb. 1, 8:00 p.m.']
    for i in range(3):
        print('%s:' % model_str[i])
        total_time[i] = (max(time_post[i]) - min(time_post[i])) / 3600 #total hours

        num_tweets_hour = np.zeros(int(total_time[i])+1)
        num_retweets_hour = np.zeros(int(total_time[i])+1)
        num_followers_hour = np.zeros(int(total_time[i])+1)
        maximum_follower = np.zeros(int(total_time[i])+1)
        favourites_count = np.zeros(int(total_time[i])+1)
        num_ranking_scores = np.zeros(int(total_time[i])+1)
        maximum_retweets = np.zeros(int(total_time[i])+1)

        min_time = min(time_post[i])
        with open(path+files) as ifile:
            for row in ifile:
                test = json.loads(row)
                index = int((test['citation_date']-min_time)/3600)
                if((index<0) | (index>int(total_time[i]))):
                    continue
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
        features = features.T
        features = features[:-1, :]
        outputs = num_tweets_hour[1:]
#         outputs = np.append(outputs, [0])
        
        mae_model1 = []
        mae_model2 = []
        mae_model3 = []
        for train_index, test_index in KFold(n_splits=10, random_state=0, shuffle=True).split(features):
            features_train, features_test = features[train_index], features[test_index]
            outputs_train, outputs_test = outputs[train_index], outputs[test_index]

            #fit the model
            clf1 = LinearRegression()
            clf1.fit(features_train, outputs_train)
            predicted_values1 = clf1.predict(features_test)
            true_values = outputs_test
            temp = mean_absolute_error(true_values, predicted_values1)
            mae_model1.append(temp)
#             print('mae is %f' %temp)
            
            clf2 = KNeighborsRegressor()
            clf2.fit(features_train, outputs_train)
            predicted_values2 = clf2.predict(features_test)
            true_values = outputs_test
            temp = mean_absolute_error(true_values, predicted_values2)
            mae_model2.append(temp)
#             print('mae is %f' %temp)

            clf3 = RandomForestRegressor()
            clf3.fit(features_train, outputs_train)
            predicted_values3 = clf3.predict(features_test)
            true_values = outputs_test
            temp = mean_absolute_error(true_values, predicted_values3)
            mae_model3.append(temp)
#             print('mae is %f' %temp)
            
        print('  average mae (linear) is %f' %np.mean(mae_model1))
        print('  average mae (knn) is %f' %np.mean(mae_model2))
        print('  average mae (random forest) is %f \n' %np.mean(mae_model3))