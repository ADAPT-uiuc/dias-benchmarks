#!/usr/bin/env python
# coding: utf-8

# ## This notebook analyses growth of startsup's in India during 2016 to 2022.

# # Import Libraries

# In[1]:


import numpy as np 
import os
# STEFANOS: Conditionally import Modin Pandas
import pandas as pd


# from wordcloud import WordCloud
import string


# # Visualisations

# In[2]:


df = pd.read_csv('input/indian-startup-recognized-by-dpiit/Startup_Counts_Across_India.csv')
df.head().style.background_gradient(cmap='coolwarm')


# # -- STEFANOS -- Replicate Data

# In[3]:


factor = 2500
df = pd.concat([df]*factor, ignore_index=True)
# df.info()


# ## Industry Vs Year

# In[4]:


# STEFANOS: Disable plotting
# fig = px.scatter(df,x="Industry", y="Year", size="Count", color="Count",template='plotly_dark', title="Industry Vs Year")
# fig.show()


# ## Year wise Startup Growth

# In[5]:


# fig, ax = plt.subplots(1,1, figsize=(15, 6))
df_year = df['Year'].value_counts().sort_index()
# STEFANOS: Disable plotting
# ax.bar(df_year.index, df_year, width=0.55,linewidth=0.7, color = 'purple')
# for i in df_year.index:
#     ax.annotate(f"{df_year[i]}",xy=(i, df_year[i] + 100),
#                    va = 'center', ha='center')
# ax.set_ylim(0, 1300)    
# fig.text(0.1, 0.95, "Growth of Startup's from 2016-2022", fontsize=15, fontweight='bold')    

# ax.grid(axis='y', linestyle='-', alpha=0.4)  


# ## Various Industries and Its Count

# In[6]:


df.groupby('Industry').size().sort_values(ascending=False).to_frame().style.background_gradient(cmap='coolwarm')


# ## Top 20 Startup Industries from 2016

# In[7]:


# fig, ax = plt.subplots(1,1, figsize=(20, 6))
df_ind = df['Industry'].value_counts().iloc[:20]

# STEFANOS: Disable plotting
# ax.bar(df_ind.index, df_ind, width=0.55,linewidth=0.7, color = 'pink')
# for i in df_ind.index:
#     ax.annotate(f"{df_ind[i]}",xy=(i, df_ind[i] + 20),va = 'center', ha='center')
# ax.set_ylim(0, 220)    
# fig.text(0.1, 0.95, "Top 20 Startup Industries from 2016", fontsize=15, fontweight='bold')    
# plt.xticks(rotation=90)
# ax.grid(axis='y', linestyle='-', alpha=0.4)  


# ## Top 20 States with Max Startup's

# In[8]:


# fig, ax = plt.subplots(1,1, figsize=(20, 6))
X = df['State'].value_counts().iloc[:20]

# STEFANOS: Disable plotting
# ax.bar(X.index, X, width=0.55,linewidth=0.7, color = 'lightblue')
# for i in X.index:
#     ax.annotate(f"{X[i]}",xy=(i, X[i] + 50),
#                    va = 'center', ha='center')
# ax.set_ylim(0, 400)    
# fig.text(0.1, 0.95, "Top 20 States Having maximum Start'ups from 2016", fontsize=15, fontweight='bold')    
# plt.xticks(rotation=90)
# ax.grid(axis='y', linestyle='-', alpha=0.4)  


# ## AI related Startup's from 2016-2022

# In[9]:


ds_list=['Internet of Things','AI','Robotics','Analytics','Computer Vision']
ds_df = df[df['Industry'].isin(ds_list)]


# In[10]:


# STEFANOS: Disable plotting
# sns.set_style('whitegrid')
# sns.catplot(x='Year', hue = 'Industry', kind='count', data=ds_df,palette="Set3", height=8.27, aspect=11.7/8.27);
# plt.show()

