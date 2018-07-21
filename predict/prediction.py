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
    checked_data=False
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
        if (dataset.isnull().sum().sum() == 0):
            numerical=false
            for i in predic_columns:
                if(is_numeric_dtype(dataset[i])):
                    numerical=true
                else:
                    numerical=false
                    break
            if numerical:
                return "valid"
            else:
                return "not vlaid"
        else: 
            return "not vlaid"
        
        
    def clean_data():
        #delete null rows 
        # return deleted rows and the new dataset after delete rows
        data_state  = check_data()
        deleted
        if data_state == "not vlaid":
            #delet nill row from dataset
            deleted=dataset[dataset==None]
            dataset = dataset[dataset!=None]
            deleted.append(dataset.columns[dataset.isnull().any()])
            dataset = dataset.dropna(axis = 0)
           
        #return the dataset after cleaning
       return dataset,deleted

    def encode(data,column):
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
    
        
    def prepare():
        #split data to training and test 
        # return boolen "true "if done "false " if not
        list_of_encode=dict()
        for col in predict_columns :
            if ~is_numeric_dtype(dataset[col]):
                en_code=encode(dataset,col)
                write(col,en_code)
        from sklearn.cross_validation import train_test_split
        train=dataset.sample(frac=0.8,random_state=1)
        test=dataset.loc[~ dataset.index.isin(train.index)]
        
        
    def best_predict():
        #call all predict functions if  checked_data is true 
        # choose least error "best algorithm for data "
        # return summary of the best predict 
        if check_data()=="valid":
            linear_reg()
            RF_reg()
            ploynomial_reg()
            return pre_dict
        else:
            print("clean ur data and prepare first")
            
        
    def linear_reg():
        #return error
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error
        pre_dict=LinearRegression()
        pre_dict.fit(train[columns],train[target])
        prediction=pre_dict.predict(test[columns])
        error=mean_squared_error(prediction,test[target])

        
        
    def RF_reg():
        #return error
        #random forest
        from sklearn.ensemble import RandomForestRegressor
        rfr=RandomForestRegressor(n_estimators=100,min_samples_leaf=10,random_state=1)
        rfr.fit(train[columns],train[target])
        pre=rfr.predict(test[columns])
        error_=mean_squared_error(pre,test[target])
        if(error_<error):
            pre_dict=rfr
            error=error_
        
        
    def ploynomial_reg():
        #return error 
        # ploynomial regression with its all possible degrees and return the best one 
        ploy_error=100
        lin_regressor
        m=dataset.shape[0]
        m_error
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
            error=m_error
        
        
        
    def predict(varibles):
        #take array of paramters country year job title 
        # return the predict "" count of jobs of this job "" 
        
        
        
        
        
