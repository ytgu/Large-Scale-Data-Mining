"""
Generate stats.txt in which each location is used as the key in the created dictionary.
The content of each item in the dictionary is a count number, which counts the frequency of each location

author: Haitao Wang
date: 3/8/2018
"""

import json
import os

path = os.path.dirname(os.path.realpath(__file__))
data = []
with open(path+'/data/tweets_#superbowl.txt') as f:
    for line in f:
        data.append(json.loads(line))

loc_dict = {}
for dat in data:
    if dat['tweet']['user']['location'] not in loc_dict:
        loc_dict[dat['tweet']['user']['location']] = 1
    else:
        loc_dict[dat['tweet']['user']['location']] += 1

with open(path+'/data/stats.txt', 'w') as f:
    f.write(json.dumps(loc_dict))
print 'number of locations:'
print len(loc_dict)
print 'number of tweets:'
print len(data)



