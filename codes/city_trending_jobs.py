# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 15:02:12 2018

@author: Sara
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


dataFrame =[]
specialty=list()
rate=list()
def count (filepath,city):
    dataFrame=pd.read_csv(filepath)
    newdata=dataFrame[dataFrame.job_location == city]
    newdata = newdata[newdata.job_date == 2018]
    return newdata
    
new = count(r'C:\Users\sara\Desktop\jobDataset_All ver.2.csv', 'Makkah')        
def encode(data,column):  
   col= dict()
   for q in range(len(data)):
       x = data.iloc[q][column]
       if not data.iloc[q][column] in col:
           col.update({x:1})
       else:
            col[x]=col[x]+1    
   return col

countt=encode(new,'job_specialty')
sorted_by_value = sorted(countt.items(), key=lambda kv: kv[1],reverse=True)
#print (sorted_by_value)
c = 0
for key, value in sorted_by_value:   
    if c < 10:
        specialty.append(key)
        rate.append(value)
        c = c+1
        



#https://medium.com/python-pandemonium/data-visualization-in-python-bar-graph-in-matplotlib-f1738602e9c4
def plot_bar_x():
    # this is for plotting purpose
    index = np.arange(len(specialty))
    plt.bar(index, rate, color = 'cgmbyr')
    plt.xlabel('Jobs', fontsize=5)
    plt.ylabel('Growth rate', fontsize=5)
    plt.xticks(index, specialty, fontsize=10, rotation=90)
    plt.title('Trending jobs')
    plt.show()
    
    
plot_bar_x()   