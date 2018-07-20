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
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_string_dtype

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
        if (dataset.isnull().sum().sum() == 0) and is_numeric_dtype(dataset['job_date']) and is_string_dtype(dataset['job_location']) and is_string_dtype(dataset['job_title']): #ceck null row (AND) check column type:
            return "vlaid"
        else: 
            return "not vlaid"
        
        
    def clean_data():
        #delete null rows 
        # return deleted rows and the new dataset after delete rows
        data_state  = check_data()
        if data_state == "not vlaid":
            #delet nill row from dataset
            dataset = dataset[dataset!=None]
            dataset = dataset.dropna(axis = 0)
            # convert column type
            dataset.job_date = dataset.job_date.astype(int)
            dataset.job_date = dataset.job_location.astype(str)
            dataset.job_date = dataset.job_title.astype(str)
        #return the dataset after cleaning
        return dataset
        
    def prepare():
        #split data to training and test 
        # return boolen "true "if done "false " if not 
        from sklearn.cross_validation import train_test_split
        train=dataset.sample(frac=0.8,random_state=1)
        test=dataset.loc[~ dataset.index.isin(train.index)]
        
        
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
        ploy_error=100
        lin_regressor
        m=dataset.shape[0]
        for i in range (1:m):
            lin_regressor = LinearRegression()
            poly = PolynomialFeatures(i)
            X_transform = poly.fit_transform(train[predict_columns])
            lin_regressor.fit(X_transform,y_train[target]) 
            y_preds = lin_regressor.predict(test[predict_columns])
            m_error=mean_squared_error(y_preds,test[target])
            if(m_error+0.0001<ploy_error):
                ploy_error=m_error
            else:
                break
        if(ploy_error<error):
            pre_dict=lin_regressor
            
        
        
        
    def predict(varibles):
        #take array of paramters country year job title 
        # return the predict "" count of jobs of this job "" 
        
        
        
        
        