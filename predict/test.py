import pandas as pd 
from prediction import prediction_Class





dataset=pd.read_csv("../data/jobDataset_All.csv")
dataset=dataset[dataset.job_specialty=="Accounting And Auditing"]
pre_cols=[""]
ma=dict()
for i in dataset:
    if ~ i.job_dates in ma:
        ma.update(i.job_dates,1)
    else:
        ma[i.job_dates]+=1

prediction_Class p (dataset,)