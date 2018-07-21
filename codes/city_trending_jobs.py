# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 15:02:12 2018

@author: Sara
"""
import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd

count_rows=  sum(1 for row in csv.reader( open(r'C:\Users\sara\Desktop\WM\Job-prediction\data\csDataset.csv')))
#print (count_rows-1)


dataFrame =[]
date = '2018'

def count_city(filepath, city):
    count = 0
    with open(filepath, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['job_location'].lower() in city.lower() and row['job_date'] in '2018':
                count = count +1
                dataFrame.append(row['job_title'])
    return count

def count (filepath,city):
    dataFrame=pd.read_csv(filepath)
    newdata=dataFrame[dataFrame.job_location == city]
    newdata = newdata[newdata.job_date == 2018]
    return newdata
    
new=count(r'C:\Users\sara\Desktop\WM\Job-prediction\data\jobDataset_All.csv', 'Makkah')        

def encode(data,column):
   
   col=dict()
   for q in range(len(data)):
       x=data.iloc[q][column]
       if not data.iloc[q][column] in col:
           col.update({x:1})
       else:
            col[x]=col[x]+1    
   return col
countt=encode(new,'job_specialty')
print(countt)
speci=list()
cou=list()
for key, value in countt.items():
    speci.append(key)
    cou.append(value)
    

print(speci)
print(cou)


#https://medium.com/python-pandemonium/data-visualization-in-python-bar-graph-in-matplotlib-f1738602e9c4
def plot_bar_x():
    # this is for plotting purpose
    index = np.arange(len(speci))
    plt.barh(index, cou)
    plt.ylabel('Jobs', fontsize=5)
    plt.xlabel('Growth rate', fontsize=5)
    plt.yticks(index, speci, fontsize=5, rotation=30)
    plt.title('Trending jobs')
    plt.show()
    
    
plot_bar_x()   