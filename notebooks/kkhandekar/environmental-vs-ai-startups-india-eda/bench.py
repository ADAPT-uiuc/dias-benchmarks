#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import os
# STEFANOS: Conditionally import Modin Pandas
import pandas as pd




# In[ ]:


def hash_anything(obj):
    import pandas.util
    if isinstance(obj, pd.DataFrame):
        return pandas.util.hash_pandas_object(obj, index=False).to_numpy().data
    elif isinstance(obj, np.ndarray):
        return obj.data
    elif isinstance(obj, list):
        return str(obj).encode()
    else:
        return str(obj).encode()


def hash_dataframe(df):
    import xxhash
    h = xxhash.xxh64()
    for column in df.round(6).columns:
        h.update(hash_anything(df[column]))
    return h.digest()


# In[2]:


# load & cleanup
file = os.path.abspath('') + '/input/indian-startup-recognized-by-dpiit/Startup_Counts_Across_India.csv'
df = pd.read_csv(file)


# # -- STEFANOS -- Replicate Data

# In[3]:


factor = 3000
df = pd.concat([df]*factor)
# df.info()


# In[4]:


df.drop('S No.',axis=1,inplace=True)
df.dropna(inplace=True)
df.reset_index(inplace=True,drop=True)

#view
df.head()


# In[5]:


# %%time
# Industry sub-categories for environmental & AI startups
env = ['Agriculture','Green Technology','Renewable Energy','Waste Management']
ai = ['AI','Robotics','Computer Vision']

# combined df - environmental & AI startups only
df_ea = df.loc[(df['Industry'].isin(env)) | (df['Industry'].isin(ai))].reset_index(drop=True,inplace=False)

# custom function to set Main Industry
def set_MainIndustry(ind):
    if ind in env:
        return 'ENV'
    else:
        return 'AI'

# adding a new column
df_ea['MainIndustry'] = df_ea.Industry.apply(lambda x: set_MainIndustry(x))

# basic stats
print(f"A total of {df_ea.shape[0]} startups were started in India between 2016 & 2022, out of which {df_ea.groupby('MainIndustry').size()['ENV']} are environmental related & {df_ea.groupby('MainIndustry').size()['AI']} are AI startups.")


# In[ ]:


print(hash_dataframe(df_ea))


# # -- STEFANOS -- Disable the rest of the code because it's plotting
