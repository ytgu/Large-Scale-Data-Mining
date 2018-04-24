
# coding: utf-8



import numpy as np
import json
import matplotlib.pyplot as plt



path = 'tweet_data/'
hashtags = ['tweets_#gohawks.txt','tweets_#gopatriots.txt','tweets_#nfl.txt','tweets_#patriots.txt',
            'tweets_#sb49.txt', 'tweets_#superbowl.txt']

for files in hashtags:
    time_post = []
    num_retweets = []
    num_followers = []
    
    with open(path+files) as txtfile:
        for row in txtfile:
            data = json.loads(row)
            time_post.append(data['citation_date'])
            num_retweets.append(data['metrics']['citations']['total'])
            num_followers.append(data['author']['followers'])
    txtfile.close()
        
    total_time = (max(time_post) - min(time_post)) / 3600 #total hours
    min_time = min(time_post)
    total_user = len(num_followers)
    total_tweets = len(num_retweets)
    total_retweets = sum(num_retweets)
    total_followers = sum(num_followers)
        
    print('=========================================================================')
    print('hashtage is: %s' %files)
    print('Average number of tweets per hour is: %f' %(total_tweets / total_time))
    print('Average number of followers of users posting the tweets is: %f' %(total_followers / total_user))
    print('Average number of retweets is: %f' %(total_retweets / total_tweets))
    print('=========================================================================')
    
    if files in ('tweets_#nfl.txt', 'tweets_#superbowl.txt'):
        num_tweets_hour = np.zeros(int(total_time)+1)                  
        with open(path+files) as ifile:
            for row in ifile:
                test = json.loads(row)
                index = int((test['citation_date']-min_time)/3600)
                num_tweets_hour[index] = num_tweets_hour[index]+1      
        ifile.close()
        x_axis = []
        for k in range(0, num_tweets_hour.shape[0]):
            x_axis.append(k)
        fig, ax = plt.subplots(figsize=(12,8))
        ax.bar(x_axis,num_tweets_hour)
        plt.title('Number of Tweets per Hour')
        plt.xlabel('time slot per hour')
        plt.ylabel('numbers of tweets')
        plt.show()  

