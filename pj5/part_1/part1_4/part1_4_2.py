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

allfeatures = {0: np.zeros((1, 7)), 1: np.zeros((1, 7)), 2: np.zeros((1, 7))}
alloutputs = {0: [0], 1: [0], 2: [0]}
for files in hashtags:
    #load the data
    time_post = [[], [], []]
    total_time = [0, 0, 0]
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
        for files in hashtags:
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
        allfeatures[i] = np.row_stack((allfeatures[i], features))
        alloutputs[i] = np.concatenate((alloutputs[i], outputs))

for i in range(3):
    print('%s:' % model_str[i])
    mae_model1 = []
    mae_model2 = []
    mae_model3 = []

    clf3 = RandomForestRegressor()
    allfeatures[i] = allfeatures[i][1:, :]
    alloutputs[i] = alloutputs[i][1:]
    clf3.fit(allfeatures[i], alloutputs[i])
    predicted_values3 = clf3.predict(allfeatures[i])
    true_values = alloutputs[i]
    temp = mean_absolute_error(true_values, predicted_values3)
    mae_model3.append(temp)
#             print('mae is %f' %temp)

#     print('  average mae (linear) is %f' %np.mean(mae_model1))
#     print('  average mae (knn) is %f' %np.mean(mae_model2))
    print('  average mae (random forest) is %f \n' %np.mean(mae_model3))
    #plot the result
    fig = plt.figure()
    plt.plot(true_values, predicted_values3,'ro',linewidth=0.1)
    plt.plot([alloutputs[i].min(),alloutputs[i].max()],[alloutputs[i].min(),alloutputs[i].max()], '--', color='blue')
    plt.xlabel('True values')     
    plt.ylabel('Predicted values')
    plt.title('Predicted Values VS. True Values For RF')
    plt.show() 