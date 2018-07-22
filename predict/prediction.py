import sys
import pandas
import sklearn
import seaborn
import matplotlib
import os
import numpy as np
import matplotlib.pylab as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_string_dtype

from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline


class prediction_Class:
    


    dataset="" 

    predict_columns=""
    target=""
    #true or false based on check_data()
    checked_data=False
    # for error of the predict algorithm 
    error="" 
    #the best algorithm
    pre_dict=""
    #train variable
    train=""
    #test
    test=""
    
    
    
    def __init__(self,datase,predic_columns,targe):
        #initialization of data
        self.dataset=datase
        self.predict_columns=predic_columns
        self.target=targe
    
    
    
    def check_data(self):
        #check validation of data
        # make checked_data true for valid or false for not valid
        #return"valid or not valid "
        if (self.dataset.isnull().sum().sum() == 0):
            numerical=False
            for i in self.predict_columns:
                if(is_numeric_dtype(self.dataset[i])):
                    numerical=True
                else:
                    numerical=False
                    break
            if numerical:
                return "valid"
            else:
                return "not vlaid"
        else: 
            return "not vlaid"
        
        
    def clean_data(self):
        #delete null rows 
        # return deleted rows and the new dataset after delete rows
        data_state  = self.check_data()
       
        if data_state == "not vlaid":
            #delet nill row from dataset
            self.dataset =self.dataset.dropna(axis = 0)
           
        #return the dataset after cleaning
        return self.dataset

    def encode(self,data,column):
        i=0
        col=dict()
        for q in range(len(data.index)):
            x=data.iloc[q][column]
            if not data.iloc[q][column] in col:
                col.update({x:i})
            
                i+=1
        data[column]=data[column].map(col)
    
        return col

    def write(filename,my_dict):
        with open(filename, 'w') as f:
            [f.write('{0},{1}\n'.format(key, value)) for key, value in my_dict.items()]
    
        
    def prepare(self):
        #split data to training and test 
        # return boolen "true "if done "false " if not
        
        for col in self.predict_columns :
            if (str(self.dataset[col].dtype)!='int64'):
                en_code=self.encode(self.dataset,col)
                self.write(col,en_code)
        from sklearn.cross_validation import train_test_split
        self.train=self.dataset.sample(frac=1,random_state=None)
        self.train=self.train.sort_values('date')
        #self.test=self.dataset.loc[~ self.dataset.index.isin(self.train.index)]
        self.test=self.train
        
    def best_predict(self):
        #call all predict functions if  checked_data is true 
        # choose least error "best algorithm for data "
        # return summary of the best predict 
        if self.check_data()=="valid":
            plt.plot(self.train[self.predict_columns],self.train[self.target])
            plt.show()
          
            self.linear_reg()
            self.RF_reg()
            self.ploynomial_reg()
            return self.pre_dict
        else:
            print("clean ur data and prepare first")
            
        
    def linear_reg(self):
        #return error
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error
        pre_dict=LinearRegression()
        pre_dict.fit(self.train[self.predict_columns],self.train[self.target])
        prediction=pre_dict.predict(self.test[self.predict_columns])
        self.error=mean_squared_error(prediction,self.test[self.target])

        
        
    def RF_reg(self):
        #return error
        #random forest
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error
        rfr=RandomForestRegressor(n_estimators=100,min_samples_leaf=10,random_state=0)
        rfr.fit(self.train[self.predict_columns],self.train[self.target])
        pre=rfr.predict(self.test[self.predict_columns])
        error_=mean_squared_error(pre,self.test[self.target])
        if(error_<self.error):
            self.pre_dict=rfr
            self.error=error_
        
        
    def ploynomial_reg(self):
        #return error 
        # ploynomial regression with its all possible degrees and return the best one 
        ploy_error=1000000000000000000000
        from sklearn.metrics import mean_squared_error
        lin_regressor=""
        m=self.dataset.shape[0]
        m_error=""
        for i in range(1,m):
            lin_regressor = LinearRegression()
            poly = PolynomialFeatures(i)
            X_transform = poly.fit_transform(self.train[self.predict_columns])
           
            lin_regressor.fit(X_transform,self.train[self.target]) 
            y_transform = poly.fit_transform(self.test[self.predict_columns])
            y_preds = lin_regressor.predict(y_transform)
            plt.plot(self.train[self.predict_columns], lin_regressor.predict(X_transform),color='g')
            m_error=mean_squared_error(y_preds,self.test[self.target])
            if(m_error+0.0001<ploy_error):
                ploy_error=m_error
            
       
        if(ploy_error<self.error):
            self.pre_dict=lin_regressor
            self.error=m_error

    
        
        
    #def predict(varibles):
        #take array of paramters country year job title 
        # return the predict "" count of jobs of this job "" 
      
