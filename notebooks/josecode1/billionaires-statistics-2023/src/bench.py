#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import os
if "IREWR_WITH_MODIN" in os.environ and os.environ["IREWR_WITH_MODIN"] == "True":
    # STEFANOS: Import Modin Pandas
    import os
    os.environ["MODIN_ENGINE"] = "ray"
    import ray
    ray.init(num_cpus=int(os.environ['MODIN_CPUS']), runtime_env={'env_vars': {'__MODIN_AUTOIMPORT_PANDAS__': '1'}})
    import modin.pandas as pd
else:
    # STEFANOS: Import regular Pandas
    import pandas as pd
# import seaborn as sns 
import matplotlib.pyplot as plt

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


def main():
    data = pd.read_csv("../input/Billionaires Statistics Dataset.csv", index_col="rank")


    # In[3]:


    factor = 800
    data = pd.concat([data]*factor, ignore_index=True)
    data.info()


    # In[4]:


    data.head(5)


    # In[5]:


    data.describe()


    # In[6]:


    country_names =data["country"].value_counts() #List of how many billionaires there are in the country


    # In[7]:


    country_names 


    # > > > > >  ****10 countries with the most billionaires****

    # In[8]:


    data_100 = data.loc[:100,["finalWorth","category","country"]]


    # In[9]:


    data_100_category = data_100["category"].value_counts()


    # > > > > > ****Most Richest 100 person in the world and their categories****

    # In[10]:


    data_usa = data[data["country"]== "United States"] #We focus on billionaires based in United States


    # In[11]:


    data_usa_category = data_usa["category"].value_counts()
    data_usa_category


    # > > > > > ****All Billionaires in the USA and their categories****

    # In[12]:


    data_usa_city = data_usa["city"].value_counts()
    print(data_usa_city)
    data_usa_city.info()


    # 268 city exist, I'll try .head(20)

    # > > > > > **The 20 cities has most billionaires in usa**

    # In[13]:


    data.head(5)


    # In[14]:


    category_list = list(data.category.unique())
    category_list


    # In[15]:



    data.finalWorth = data.finalWorth.astype(float)
    worth_average = []
    ### SLOW TO BE OPTIMIZED START ###
    for i in category_list:
        x = data[data['category']==i]
        worth_ratio = sum(x.finalWorth)/len(x)
        worth_average.append(worth_ratio)
    ### SLOW TO BE OPTIMIZED END ###
    data2 = pd.DataFrame({'category_list': category_list,'worth_average':worth_average})
    new_index = (data2['worth_average'].sort_values(ascending=False)).index.values
    new_data = data2.reindex(new_index)


    # In[16]:


    data3 =data.dropna(axis="index",how="any",inplace=False) # ı created new df because so many nan values was exist


    # In[17]:


    data3.head() 


    # In[18]:


    data3.info() #in the new df we have 238 rows for each column


    # In[19]:


    data3.country.unique()  # and 238 billionaires in usa, others deleted.


    # **kernel will continue...**
