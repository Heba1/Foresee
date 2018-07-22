
# coding: utf-8

# In[77]:

import pandas as pd 
from prediction import prediction_Class






# In[78]:


dataset=pd.read_csv("../data/jobDataset_All.csv")
dataset=dataset[dataset.job_specialty=="Accounting And Auditing"]


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


# In[89]:


print()

# In[ ]:



