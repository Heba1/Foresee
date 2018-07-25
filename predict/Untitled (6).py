
# coding: utf-8

# In[77]:

import pandas as pd 
from prediction1 import prediction_Class
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pylab as plt




# In[78]:


dataset=pd.read_csv("../data/jobDataset_All.csv")
dataset=dataset[dataset.job_specialty=="Accounting And Auditing"]
dataset=dataset[dataset.job_date>"2013"]

# In[79]:

dataset


# In[80]:

ma=dict()
for i in range(len(dataset)):
    x=dataset.iloc[i]
    if not x.job_date in ma:
        ma.update({x.job_date:1})
    else:
        ma[x.job_date]+=1



# In[81]:

date=list()
count=list()

for key, value in ma.items():
        date.append(key)
        count.append(value)


# In[82]:

n_data=pd.DataFrame({"date":date,"count":count})


# In[83]:

n_data



# In[84]:

pre_cols=["date"]
target="count"
n_data["date"]=pd.to_numeric(n_data["date"], errors='coerce')
n_data["count"]=pd.to_numeric(n_data["count"], errors='coerce')
n_data.sort_values('date')

p =prediction_Class(n_data,pre_cols,target)


# In[87]:
n=p.clean_data()
n=p.prepare()
n=p.best_predict()
da=list()
da.append(2017)

a=pd.DataFrame({"date":da})



y=p.poly.transform(a.reset_index().values)
res=n.predict(y)
plt.cla()
plt.clf()
plt.scatter(a["date"].reset_index().values,p.pre_dict.predict(y),color="r")
plt.axis([2013,2020,0,1000])
plt.show()
# In[89]:

print(p.train)
print(p.test)
print(p.error)

# In[ ]:



