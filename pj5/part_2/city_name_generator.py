"""
Generate the search names for most frequent cities in either Washington state or Massachusetts state

author: Haitao Wang
Date: 3/8/2018
"""

import os
import json
import re
import time

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

reg_wa = ['^(.*), wa$', '^(.*),wa$', '^(.*), wa[.*]', '^(.*),wa[^a-z]']#['^([a-z]+), wa$', '^([a-z]+),wa$', '^([a-z]+), wa[^a-z]', '^([a-z]+),wa[^a-z]']
reg_ma = ['^(.*), ma$', '^(.*),ma$', '^(.*), ma[.*]', '^(.*),ma[^a-z]']#['^([a-z]+), ma$', '^([a-z]+),ma$', '^([a-z]+), ma[^a-z]', '^([a-z]+),ma[^a-z]']

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
        city_ma.append(city.encode('ascii', 'ignore'))
print city_wa
print city_ma
