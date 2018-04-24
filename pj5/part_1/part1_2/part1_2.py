
# coding: utf-8




import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from math import sqrt




path = 'tweet_data/'
hashtags = ['tweets_#gohawks.txt','tweets_#gopatriots.txt','tweets_#nfl.txt','tweets_#patriots.txt',
            'tweets_#sb49.txt', 'tweets_#superbowl.txt']

def rmse(y, pred):
    return sqrt(mean_squared_error(y, pred))

for files in hashtags:
    time_post = []
    num_retweets = []
    num_followers = []
    
    #load the data
    with open(path+files) as txtfile:
        for row in txtfile:
            data = json.loads(row)
            time_post.append(data['citation_date'])
            #num_retweets.append(data['metrics']['citations']['total'])
            #num_followers.append(data['author']['followers'])
    txtfile.close()
    total_time = (max(time_post) - min(time_post)) / 3600 #total hours
    min_time = min(time_post)
          
    num_tweets_hour = np.zeros(int(total_time)+1)
    num_retweets_hour = np.zeros(int(total_time)+1)
    num_followers_hour = np.zeros(int(total_time)+1)
    maximum_follower = np.zeros(int(total_time)+1)
    time_of_day = np.zeros(int(total_time)+1)
        
    with open(path+files) as ifile:
        for row in ifile:
            test = json.loads(row)
            index = int((test['citation_date']-min_time)/3600)
            num_tweets_hour[index] = num_tweets_hour[index]+1
            num_retweets_hour[index] = num_retweets_hour[index] + test['metrics']['citations']['total']
            num_followers_hour[index] = num_followers_hour[index] + test['author']['followers']
            if( test['author']['followers'] >= maximum_follower[index]):
                maximum_follower[index] = test['author']['followers']
            time_of_day[index] = index % 24
    ifile.close()
        
    #prepocess the data into dataset
    features = np.array([num_tweets_hour, num_retweets_hour, num_followers_hour, maximum_follower, time_of_day])
    features = features.T
    outputs = num_tweets_hour[1:]
    outputs = np.append(outputs, [0])

    #fit the model
    clf = LinearRegression()
    clf.fit(features, outputs)
    predicted_values = clf.predict(features)
    true_values = outputs
    
    #calculate the output
    print('=========================================================================')
    print('hashtage is: %s \n' %files)
    
    #calculate the RMSE
    rmse_model = rmse(true_values, predicted_values)
    print('rmse is %f \n' %rmse_model)
    
    #print the statistic summary
    results = sm.OLS(outputs, features).fit() 
    print(results.summary())

    #plot the result
    fig = plt.figure()
    plt.plot(true_values, predicted_values,'ro',linewidth=0.1)
    plt.plot([outputs.min(),outputs.max()],[outputs.min(),outputs.max()], '--', color='blue')
    plt.xlabel('True values')     
    plt.ylabel('Predicted values')
    plt.title('Predicted Values VS. True Values For Linear Regression')
    plt.show()
    print('=========================================================================')    



