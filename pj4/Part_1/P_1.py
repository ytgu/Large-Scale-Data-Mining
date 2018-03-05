# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 15:50:39 2018

@author: ht
"""

import csv
import colorsys
import matplotlib.pyplot as plt
#read data
path = '/users/ht/desktop/EE219/proj_4/'
data = []
work_flow_list = []
with open(path+'data/network_backup_dataset.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        data.append((int(row['Week #']), row['Day of Week'],
                         int(row['Backup Start Time - Hour of Day']),row['Work-Flow-ID'],
                         row['File Name'],float(row['Size of Backup (GB)']),
                         int(row['Backup Time (hour)'])))
        work_flow_list.append(row['Work-Flow-ID'])
work_flow_list = set(work_flow_list)
backup_size_list = {}
subdata = {}
for term in work_flow_list:
    backup_size_list[term] = []
    subdata[term] = []

for d in data:
    subdata[d[3]].append(d)
for key in subdata.keys():
    sd = subdata[key]
    backup_size = sd[0][5]
    for idx in range(1,len(sd)): 
        if (sd[idx][0] == sd[idx-1][0]) & (sd[idx][1] == sd[idx-1][1]):
            backup_size += sd[idx][5]
        else:
            backup_size_list[key].append(backup_size)
            backup_size = sd[idx][5]
    backup_size_list[key].append(backup_size)

##PART 1a
N = len(subdata)
index = range(N)
HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
keys = sorted(backup_size_list.keys())
day_list = range(1,len(backup_size_list[keys[0]])+1)
fig1 = plt.figure()
for i in index:
    plt.plot(day_list[:20], backup_size_list[keys[i]][:20], color=RGB_tuples[i], label=keys[i])
plt.xlabel('Day number')     
plt.ylabel('Backup size(GB)')
plt.title('Backup Size against Day Number For a Twenty-day Period')
plt.grid(color='0.7', linestyle='-', linewidth=1)
#plt.xlim([0.0, 25.0])
#plt.ylim([0.0, 1.0])
plt.legend(loc="upper right")
plt.show()
fig1.savefig(path+'fig/P1_20.png')

##PART 1b
fig2 = plt.figure()
for i in index:
    plt.plot(day_list, backup_size_list[keys[i]], color=RGB_tuples[i], label=keys[i])
plt.xlabel('Day number')     
plt.ylabel('Backup size(GB)')
plt.title('Backup Size against Day Number For the First 105-day Period')
plt.grid(color='0.7', linestyle='-', linewidth=1)
#plt.xlim([0.0, 25.0])
#plt.ylim([0.0, 1.0])
plt.legend(loc="upper right")
plt.show()
fig1.savefig(path+'fig/P1_105.png')