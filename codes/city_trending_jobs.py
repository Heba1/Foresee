# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 15:02:12 2018

@author: Sara
"""
import matplotlib.pyplot as plt
import numpy as np
import csv


count_rows=  sum(1 for row in csv.reader( open(r'C:\Users\sara\Desktop\WM\Job-prediction\data\csDataset.csv')))
#print (count_rows-1)


dataFrame =[]



def count_city(filepath, city):
    count = 0
    with open(filepath, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['job_location'].lower() in city.lower() and row['job_date'] in '2018':
                count = count +1
                dataFrame.append(row['job_title'])
    return count

count_city(r'C:\Users\sara\Desktop\WM\Job-prediction\data\csDataset.csv', 'makkah')        

def encode(data,column):
   i=0
   col=dict()
   for q in range(len(data)):
       x=data.iloc[q][column]
       if not data.iloc[q][column] in col:
           col.update({x:1})
       else:
            col[x]=col[x]+1    
   return col
print (encode(dataFrame,'job_title'))






#https://medium.com/python-pandemonium/data-visualization-in-python-bar-graph-in-matplotlib-f1738602e9c4
label = ['Computer Science','kkkk','ggggg','eeee','aaaa','ssss','dddd','fff']
no_movies = [count_city(r'C:\Users\sara\Desktop\WM\Job-prediction\data\csDataset.csv',"jeddah"),]

def plot_bar_x():
    # this is for plotting purpose
    index = np.arange(len(label))
    plt.barh(index, no_movies)
    plt.ylabel('Genre', fontsize=5)
    plt.xlabel('No of Movies', fontsize=5)
    plt.yticks(index, label, fontsize=5, rotation=30)
    plt.title('Market Share for Each Genre 1995-2017')
    plt.show()
    
    
#plot_bar_x()   