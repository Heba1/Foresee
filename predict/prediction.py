import sys
import pandas as pd
import sklearn
import seaborn
import matplotlib
import os
import math
import numpy as np
import matplotlib.pylab as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_string_dtype

from sklearn.svm import SVR

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
    # the values we wanna preict 
    v_predict=""
    #name of best predict
    name_bestPredict=""
    #poly
    poly=""
    
    
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
    
        
    def prepare(self,data=dataset):
        #split data to training and test 
        # return boolen "true "if done "false " if not
        
        for col in self.predict_columns :
            if (str(data[col].dtype)!='int64'):
                en_code=self.encode(self.dataset,col)
                self.write(col,en_code)
        from sklearn.cross_validation import train_test_split
        self.train=data.sample(frac=1,random_state=None)
        self.train=self.train.sort_values('job_date')
        #self.test=self.dataset.loc[~ self.dataset.index.isin(self.train.index)]
        self.test=self.train
        
    def best_predict(self,data=dataset):
        #call all predict functions if  checked_data is true 
        # choose least error "best algorithm for data "
        # return summary of the best predict 
        if self.check_data()=="valid":
            plt.scatter(self.train[self.predict_columns],self.train[self.target])
            plt.axis([2007,2019,0,5])
            plt.show()
          
            self.linear_reg(data)
            self.RF_reg(data)
            #self.ploynomial_reg(data)
            return self.pre_dict
        else:
            print("clean ur data and prepare first")
       
        
    def algo_reg(self,data=dataset):
        count=list()
        for i in data["count"]:
            count.append(math.log(i,10))
        data["count"]=count
    def linear_reg(self,data=dataset):
        #return error
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error
        pre_dict=LinearRegression()
        pre_dict.fit(self.train[self.predict_columns],self.train[self.target])
        prediction=pre_dict.predict(self.test[self.predict_columns])
        self.error=mean_squared_error(prediction,self.test[self.target])
        self.name_bestPredict="linear"
        self.pre_dict=pre_dict
        
    def RF_reg(self,data=dataset):
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
        
        
    def ploynomial_reg(self,data=dataset):
        #return error 
        # ploynomial regression with its all possible degrees and return the best one 
        ploy_error=1000000000000000000000
        from sklearn.metrics import mean_squared_error
        lin_regressor=""
        m=data.shape[0]
        m_error=""
        i=0
        poly=""
        for i in range(1,m):
            lin_regressor = LinearRegression()
            poly = PolynomialFeatures(i)
            X_transform = poly.fit_transform(self.train[self.predict_columns].reset_index().values)
           
            lin_regressor.fit(X_transform,self.train[self.target].reset_index().values) 
            y_transform = poly.fit_transform(self.test[self.predict_columns].reset_index().values)
            y_preds = lin_regressor.predict(y_transform)

           
            
            plt.plot(self.train[self.predict_columns].reset_index().values, lin_regressor.predict(X_transform),color='g')
            plt.axis([2008,2019,0,10])

            #plt.plot(self.train[self.predict_columns].reset_index().values, lin_regressor.predict(X_transform),color='g')

            m_error=mean_squared_error(y_preds,self.test[self.target].reset_index().values)
            if(m_error+0.0001<ploy_error):
                ploy_error=m_error
            
        plt.plot(self.train[self.predict_columns].reset_index().values, lin_regressor.predict(X_transform),color='g')
        if(ploy_error<self.error):
            
            
            self.pre_dict=lin_regressor
            self.error=m_error
            self.name_bestPredict="poly"
            self.poly=i
            
            
    def get_count(self,specillaty , country="saudi"):
         new=self.dataset[self.dataset.job_specialty==specillaty]
         if country !="saudi":
             new=self.dataset[self.dataset.job_location==country]
         else:
             new["job_location"]=country
             
         city_date=dict()
         for i in range(len(new)):
            x=new.iloc[i]
            if not x.job_date in city_date:
                city_date.update({x.job_date:1})
            else:
                city_date[x.job_date]+=1
         return city_date
             
                   
            
    def predict(self,specillaty , country="saudi",date1=np.array([2018,2019,2020,2021,2022,2023])):
        #take array of paramters country year job title 
        # return the predict "" count of jobs of this job "" 
        dic=self.get_count(specillaty,country)
        date=list()
        count=list()

        for key, value in dic.items():
            date.append(key)
            count.append(value)
        n_data=pd.DataFrame({"job_date":date,"count":count})
        self.algo_reg(n_data)
        self.prepare(n_data)
        datee=pd.DataFrame({"date":date1})
        
      
        self.best_predict(n_data)
        
        y_transform=datee["date"]
        
       
        if self.name_bestPredict=="poly":
            pol = PolynomialFeatures(self.poly)
            y_transform = pol.fit_transform(y_transform.reset_index().values)
            
        y_preds = self.pre_dict.predict([[2017],[2018],[2019],[2020]])
        plt.cla()
        plt.clf()
            
        plt.scatter([[2017],[2018],[2019],[2020]], y_preds,color='r')
        plt.axis([2016,2030,0,5])
        
        return plt
            
        

# In[82]:



       
        """

        koeficienti_polinom = np.polyfit(self.train[self.predict_columns].values.ravel(), self.train[self.target].values.ravel(), 3)

        a=koeficienti_polinom[0]
        b=koeficienti_polinom[1]
        c=koeficienti_polinom[2]
    
        xval=np.linspace(np.min(self.train[self.predict_columns]), np.max(self.train[self.predict_columns]))   
            
        regression=koeficienti_polinom[0] * xval**2 + koeficienti_polinom[1]*xval +koeficienti_polinom[2] 

          
        predY = koeficienti_polinom[0] * self.train[self.predict_columns]**2 + koeficienti_polinom[1]*self.train[self.predict_columns] + koeficienti_polinom[2]   

        plt.scatter(self.train[self.predict_columns],self.train[self.target], s=20, color="blue" )      
        plt.scatter(self.train[self.predict_columns], predY, color="red")    
        plt.plot(xval, regression, color="black", linewidth=1)  
        
        
        
        
        
        
        
       
        dates=np.reshape(self.train[self.predict_columns],(len(self.train[self.predict_columns]),1))

        svr_poly=SVR(kernel="poly",C=1e3, degree=2)
        svr_poly.fit(self.train[self.predict_columns],self.train[self.target])

        plt.scatter(self.train[self.predict_columns],self.train[self.target], color="blue")
        plt.plot(self.train[self.predict_columns], svr_poly.predict(self.train[self.predict_columns]), color="red")

        plt.xlabel("Size")
        plt.ylabel("Cost")
        plt.title("prediction")
        plt.legend()

        plt.show()
        
        """
    
        
        

   


    #def predict(varibles):
        #take array of paramters country year job title 
        # return the predict "" count of jobs of this job "" 
        
      

