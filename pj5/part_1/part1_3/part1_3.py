
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
    
    #calculate the time_post
    with open(path+files) as txtfile:
        for row in txtfile:
            data = json.loads(row)
            time_post.append(data['citation_date'])

    txtfile.close()
    total_time = (max(time_post) - min(time_post)) / 3600 #total hours
    min_time = min(time_post) 
    
    
    #load the data      
    num_tweets_hour = np.zeros(int(total_time)+1)
    num_retweets_hour = np.zeros(int(total_time)+1)
    num_followers_hour = np.zeros(int(total_time)+1)
    maximum_follower = np.zeros(int(total_time)+1)
    favourites_count = np.zeros(int(total_time)+1)
    num_ranking_scores = np.zeros(int(total_time)+1)
    maximum_retweets = np.zeros(int(total_time)+1) 
        
    with open(path+files) as ifile:
        for row in ifile:
            test = json.loads(row)
            index = int((test['citation_date']-min_time)/3600)
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
                     favourites_count, maximum_retweets,  num_ranking_scores])
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
    rmse_model = rmse(true_values, predicted_values)
    print('rmse is %f \n' %rmse_model)

    results = sm.OLS(outputs, features).fit() 
    print(results.summary())

    fig = plt.figure()
    plt.plot(true_values, predicted_values,'ro',linewidth=0.1)
    plt.plot([outputs.min(),outputs.max()],[outputs.min(),outputs.max()], '--', color='blue')
    plt.xlabel('True values')     
    plt.ylabel('Predicted values')
    plt.title('Predicted Values VS. True Values For Linear Regression')
    plt.show()
    print('=========================================================================')
    
    
    
    #Draw scatter plot of top 3 features of each hashtags
    
    #top 3 features of gohwaks: num_retweets_hour, num_followers_hour, favourite_count
    if(files == 'tweets_#gohawks.txt'):
        fig = plt.figure()
        plt.plot(features[:,1],predicted_values,'ro',linewidth=0.1)
        plt.ylabel('Predicted values')     
        plt.xlabel('Number of retweets')
        plt.title('Predicted Values VS. Number of retweets(#gohwaks)')
        fig.savefig('1.3_figures/part1_3_gohawks_1.png')
        plt.show()
        
        fig = plt.figure()
        plt.plot(features[:,2],predicted_values,'ro',linewidth=0.1)
        plt.ylabel('Predicted values')     
        plt.xlabel('Number of followers')
        plt.title('Predicted Values VS. Number of followers(#gohwaks)')
        fig.savefig('1.3_figures/part1_3_gohawks_2.png')
        plt.show()
        
        fig = plt.figure()
        plt.plot(features[:,4],predicted_values,'ro',linewidth=0.1)
        plt.ylabel('Predicted values')     
        plt.xlabel('Favourite count')
        plt.title('Predicted Values VS. Favourite count(#gohwaks)')
        fig.savefig('1.3_figures/part1_3_gohawks_3.png')
        plt.show()
        
    #top 4 features of gopatriots: num_tweets_hour, num_retweets_hour, favourites_count, maximum_retweets
    elif(files == 'tweets_#gopatriots.txt'):
        fig = plt.figure()
        plt.plot(features[:,0],predicted_values,'ro',linewidth=0.1)
        plt.ylabel('Predicted values')     
        plt.xlabel('Number of tweets per hour')
        plt.title('Predicted Values VS. Number of tweets per hour(#gopatriots)')
        fig.savefig('1.3_figures/part1_3_gopatriots_1.png')
        plt.show()
        
        fig = plt.figure()
        plt.plot(features[:,1],predicted_values,'ro',linewidth=0.1)
        plt.ylabel('Predicted values')     
        plt.xlabel('Number of retweets')
        plt.title('Predicted Values VS. Number of retweets(#gopatriots)')
        fig.savefig('1.3_figures/part1_3_gopatriots_2.png')
        plt.show()
        
        fig = plt.figure()
        plt.plot(features[:,4],predicted_values,'ro',linewidth=0.1)
        plt.ylabel('Predicted values')     
        plt.xlabel('Favourites count')
        plt.title('Predicted Values VS. Favourites count(#gopatriots)')
        fig.savefig('1.3_figures/part1_3_gopatriots_3.png')
        plt.show()
        
        fig = plt.figure()
        plt.plot(features[:,5],predicted_values,'ro',linewidth=0.1)
        plt.ylabel('Predicted values')     
        plt.xlabel('Maximum retweets')
        plt.title('Predicted Values VS. Maximum retweets(#gopatriots)')
        fig.savefig('1.3_figures/part1_3_gopatriots_4.png')
        plt.show()
        
    #top 3 features of nfl: num_tweets_hour, num_ranking_scores, num_retweets_hour
    elif(files == 'tweets_#nfl.txt'):
        fig = plt.figure()
        plt.plot(features[:,0],predicted_values,'ro',linewidth=0.1)
        plt.ylabel('Predicted values')     
        plt.xlabel('Number of tweets per hour')
        plt.title('Predicted Values VS. Number of tweets per hour(#nfl)')
        fig.savefig('1.3_figures/part1_3_nfl_1.png')
        plt.show()
        
        fig = plt.figure()
        plt.plot(features[:,6],predicted_values,'ro',linewidth=0.1)
        plt.ylabel('Predicted values')     
        plt.xlabel('Ranking scores')
        plt.title('Predicted Values VS. Ranking scores(#nfl)')
        fig.savefig('1.3_figures/part1_3_nfl_2.png')
        plt.show()
        
        fig = plt.figure()
        plt.plot(features[:,1],predicted_values,'ro',linewidth=0.1)
        plt.ylabel('Predicted values')     
        plt.xlabel('Number of retweets')
        plt.title('Predicted Values VS. Number of retweets(#nfl)')
        fig.savefig('1.3_figures/part1_3_nfl_3.png')
        plt.show()
        
    #top 3 features of patriots: num_followers_hour, favourites_count, num_ranking_scores
    elif(files == 'tweets_#patriots.txt'):        
        fig = plt.figure()
        plt.plot(features[:,2],predicted_values,'ro',linewidth=0.1)
        plt.ylabel('Predicted values')     
        plt.xlabel('Number of followers')
        plt.title('Predicted Values VS. Numer of followers(#patriots)')
        fig.savefig('1.3_figures/part1_3_patriots_1.png')
        plt.show()
        
        fig = plt.figure()
        plt.plot(features[:,4],predicted_values,'ro',linewidth=0.1)
        plt.ylabel('Predicted values')     
        plt.xlabel('Favourites count')
        plt.title('Predicted Values VS.Favourites count(#patriots)')
        fig.savefig('1.3_figures/part1_3_patriots_2.png')
        plt.show()
        
        
        fig = plt.figure()
        plt.plot(features[:,6],predicted_values,'ro',linewidth=0.1)
        plt.ylabel('Predicted values')     
        plt.xlabel('Ranking scores')
        plt.title('Predicted Values VS. Ranking scores(#patriots)')
        fig.savefig('1.3_figures/part1_3_patriots_3.png')
        plt.show()
   
    #top 4 features of sb49: num_tweets_hour, num_followers_hour, favourites_count, num_ranking_scores
    elif(files == 'tweets_#sb49.txt'):
        fig = plt.figure()
        plt.plot(features[:,0],predicted_values,'ro',linewidth=0.1)
        plt.ylabel('Predicted values')     
        plt.xlabel('Number of tweets per hour')
        plt.title('Predicted Values VS. Number of tweets per hour(#sb49)')
        fig.savefig('1.3_figures/part1_3_sb49_1.png')
        plt.show()

        fig = plt.figure()
        plt.plot(features[:,2],predicted_values,'ro',linewidth=0.1)
        plt.ylabel('Predicted values')     
        plt.xlabel('Number of followers')
        plt.title('Predicted Values VS. Number of followers(#sb49)')
        fig.savefig('1.3_figures/part1_3_sb49_2.png')
        plt.show()

        fig = plt.figure()
        plt.plot(features[:,4],predicted_values,'ro',linewidth=0.1)
        plt.ylabel('Predicted values')     
        plt.xlabel('Favourites count')
        plt.title('Predicted Values VS. Favourites count(#sb49)')
        fig.savefig('1.3_figures/part1_3_sb49_3.png')
        plt.show() 
        
        fig = plt.figure()
        plt.plot(features[:,6],predicted_values,'ro',linewidth=0.1)
        plt.ylabel('Predicted values')     
        plt.xlabel('Ranking scores')
        plt.title('Predicted Values VS. Ranking scores(#sb49)')
        fig.savefig('1.3_figures/part1_3_sb49_4.png')
        plt.show()

    #top 3 features of superbowl: num_retweets_hour, num_followers_hour, maximum_follower
    elif(files == 'tweets_#superbowl.txt'):      
        fig = plt.figure()
        plt.plot(features[:,1],predicted_values,'ro',linewidth=0.1)
        plt.ylabel('Predicted values')     
        plt.xlabel('Number of retweets')
        plt.title('Predicted Values VS.Number of retweets(#superbowl)')
        fig.savefig('1.3_figures/part1_3_superbowl_1.png')
        plt.show()
        
        fig = plt.figure()
        plt.plot(features[:,2],predicted_values,'ro',linewidth=0.1)
        plt.ylabel('Predicted values')     
        plt.xlabel('Number of followers')
        plt.title('Predicted Values VS.Number of followers(#superbowl)')
        fig.savefig('1.3_figures/part1_3_superbowl_2.png')
        plt.show()     
        
        fig = plt.figure()
        plt.plot(features[:,3],predicted_values,'ro',linewidth=0.1)
        plt.ylabel('Predicted values')     
        plt.xlabel('Maximum followers')
        plt.title('Predicted Values VS. Maximum followers(#superbowl)')
        fig.savefig('1.3_figures/part1_3_superbowl_3.png')
        plt.show()
 

 