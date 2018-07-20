import sys
import pandas
import sklearn
import seaborn
import matplotlib
import os
import matplotlib.pylab as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


#class prediction
class prediction:
    dataset
    predict_columns
    target
    #true or false based on check_data()
    checked_data
    # for error of the predict algorithm 
    error 
    #the best algorithm
    pre_dict
    #train variable
    train
    #test
    test
    
    def __init__(self,datase,predic_columns,targe):
        #initialization of data
        self.dataset=datase
        self.predict_columns=predic_columns
        self.target=targe
    
    
    
    def check_data():
        #check validation of data
        # make checked_data true for valid or false for not valid
        #return"valid or not valid "
        
    def clean_data():
        
        #delete null rows 
        # return deleted rows and the new dataset after delete rows
        
    def prepare():
        #split data to training and test 
        # return boolen "true "if done "false " if not 
        
        
    def best_predict():
        #call all predict functions if  checked_data is true 
        # choose least error "best algorithm for data "
        # return summary of the best predict 
        linear_reg()
        RF_reg()
        ploynomial_reg()
        return pre_dict
        
        
    def linear_reg():
        #return error
        
        
        
     def RF_reg():
        #return error
        #random forest 
        
     def ploynomial_reg():
        #return error 
        # ploynomial regression with its all possible degrees and return the best one 
        
        
        
        
    def predict(varibles):
        #take array of paramters country year job title 
        # return the predict "" count of jobs of this job "" 
        
        
        
        
        