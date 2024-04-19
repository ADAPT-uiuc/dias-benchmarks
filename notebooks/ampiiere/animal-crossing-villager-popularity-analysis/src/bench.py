#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import os

import pandas as pd
# import seaborn as sns  # data visualization
# import matplotlib.pyplot as plt


# Input data files are available in the read-only "input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Introduction
# The goal of this project is to analyse the relationship between animal crossing new horizon villager popularity amongst the player base and certain villager attributes. 
# 
# We will be analysing the Gender,Personality, Species, and Style of a villager. 

# # Data Initilization and Cleaning

# In[2]:


vlgr_df = pd.read_csv("input/animal-crossing-new-horizons-nookplaza-dataset/villagers.csv")
popul_df = pd.read_csv("input/acnh-villager-popularity/acnh_villager_data.csv")


# # -- STEFANOS -- Replicate Data

# In[3]:


factor = 400
if "IREWR_LESS_REPLICATION" in os.environ and os.environ["IREWR_LESS_REPLICATION"] == "True":
    factor = factor//3
popul_df = pd.concat([popul_df]*factor, ignore_index=True)
vlgr_df = pd.concat([vlgr_df]*factor, ignore_index=True)
# print(popul_df.info())
# vlgr_df.info()


# In[4]:


vlgr_df.head()


# In[5]:


popul_df.head()


# ### 1. Checking for null 

# In[6]:


# vlgr_df.info()


# In[7]:


# popul_df.info()


# ### 2. Checking for mismatched names

# In[8]:


# There are some missing/non-matching names 
vlgr_df["Name"].isin(popul_df['name']).sum()


# In[9]:


# vlgr_df does not have these names...
mismatch_names = popul_df["name"][popul_df["name"].isin(vlgr_df["Name"]) == False]
mismatch_names


# In[10]:


# %%time
# Data set is small enough to pick out the same names
# Correcting names in popul_df to match vlgr_df
popul_df['name'] = popul_df['name'].replace(['OHare'],"O\'Hare")
popul_df['name'] = popul_df['name'].replace(['Buck(Brows)'],"Buck")
popul_df['name'] = popul_df['name'].replace(['Renee'],"Renée")
popul_df['name'] = popul_df['name'].replace(['WartJr'],"Wart Jr.")
popul_df['name'] = popul_df['name'].replace(['Crackle(Spork)'],"Spork")


# In[11]:


# Checking if All names match
vlgr_df["Name"].isin(popul_df['name']).sum()


# In[12]:


# drop villagers that are in popul_df but not in vlgr_df
popul_df = popul_df.drop(popul_df[popul_df["name"].isin(vlgr_df["Name"]) == False].index)


# ### 3. Merging the two Dataframes

# In[13]:


# Now that both df have same length, we can set index as names and combine the 2 dfs
popul_df.set_index('name', drop=True, inplace=True)
vlgr_df.set_index('Name', drop=True, inplace=True)


# In[14]:


combined_df = popul_df.merge(vlgr_df, left_index=True, right_index=True)


# In[15]:


# drop irrelevent columns
combined_df.drop(columns=['Furniture List', 'Filename', 'Unique Entry ID', "Wallpaper", "Flooring", "Birthday", "Favorite Song"], inplace=True)


# #### Adding a new row named overall_ranking so we may know a villager's general ranking outside of their tier

# In[16]:


combined_df.sort_values(['tier', 'rank'], inplace=True)
combined_df['overall_ranking'] = np.arange(1, len(combined_df)+1)
combined_df.insert(2, 'overall_ranking', combined_df.pop('overall_ranking'))


# #### Setting Baseline overall ranking mean to compare against

# In[17]:


overall_mean = combined_df.overall_ranking.mean()
print(f'The overall_mean is {overall_mean}, this would serve as a baseline for to compare against popularity performance of our features.')


# In[18]:


combined_df.columns


# # Exploratory Data Analysis
# As a preface, a higher overall_ranking would mean performing worse on the popularity rankings.
# ### 1. Gender

# In[19]:


combined_df['Gender'].value_counts()


# In[20]:


## -- STEFANOS: Disable plotting
# combined_df.groupby('tier').Gender.value_counts().plot.barh()
## -- STEFANOS-DISABLE-FOR-MODIN: Actually, it seems we can't even run the last part with Modin, so
## we only leave the groupby.
# combined_df.groupby('tier').Gender.value_counts()
combined_df.groupby('tier')


# For gender, there seems to be a disproporationate amount of male villagers in the lowest tier(6th tier) than female villagers, compared to other tiers. Discounting Tier 6, The number of male and female villagers are fairly even, with Male villagers having a slight lead in all tiers(except tier 6).

# In[21]:


## -- STEFANOS: Disable plotting
# plt.figure(figsize=(5, 5))
# plt.axhline(overall_mean, color='r')
# sns.boxplot(x="Gender", y='overall_ranking', data=combined_df)


# Female villagers generally perform better than Male villagers in terms of overall ranking. 

# In[22]:


pd.pivot_table(combined_df, index = 'tier', values = 'Catchphrase', columns="Gender", aggfunc='count')


# ### 2. Species

# In[23]:


# combined_df.info()


# In[ ]:





# In[24]:


combined_df.groupby('Species').mean(numeric_only=True)


# In[25]:


# creating value counts dataframe for each species type
species_ranking = combined_df.groupby('Species').mean(numeric_only=True)['overall_ranking'].to_frame().reset_index().sort_values('overall_ranking')
species_ranking


# In[26]:


## -- STEFANOS: Disable plotting
# plt.figure(figsize=(30,5))
# sns.set(font_scale=1.4)
# plt.xticks(rotation=45)
# plt.axhline(overall_mean, color='r')
# sns.scatterplot(x='Species', y="overall_ranking", data=species_ranking,label='mean overall-ranking', s=300)


# Octopus, deer, wolves, cats and Koalas are most likely to be popular; while Kangaroos, Hippos, Mouse Pigs and Gorillas are the least likely to be popular. 

# In[27]:


## -- STEFANOS: Disable plotting
# plt.figure(figsize=(30, 10))
# plt.axhline(overall_mean, color='r')
# sns.scatterplot(x="Species", y='overall_ranking', hue='tier', s=100, data=combined_df)


# Although Octopuses seem to be ranking highly in part due to the low amount of Octopuses amongst the villagers. 
# Interesting trend can be observed, there exists a ranking cap for low ranking speices, for example, none of the Gorilla villagers have a ranking lower than 200, it is heavily skewed, and not normally distributed.  Indicating a clear non-preference for certain species by the playerbase. 

# ### 3. Personality

# In[28]:


combined_df.Personality.value_counts()


# In[29]:


# creating value counts dataframe for each personality type
personality_ranking = combined_df.groupby('Personality').mean(numeric_only=True)['overall_ranking'].to_frame().reset_index().sort_values('overall_ranking')


# In[30]:


## -- STEFANOS: Disable plotting
# plt.figure(figsize=(20,5))
# sns.set(font_scale=1.4)
# plt.xticks(rotation=45)
# plt.axhline(overall_mean, color='r')
# sns.scatterplot(x='Personality', y="overall_ranking", data=personality_ranking,label='mean personality ranking', s=300)


# The playerbase seems to have a preference for Big sister, Normal, Peppy and sometimes Lazy type villagers.
# While they dislike Cranky, Jock and Snooty villagers. 

# In[31]:


## -- STEFANOS: Disable plotting
# plt.figure(figsize=(10, 10))
# plt.axhline(overall_mean, color='r')
# sns.boxplot(x="Personality", y='overall_ranking', data=combined_df)


# There seems to be a clear preference for Big Sister, Peppy and Normal Personality villagers, they have means below overall mean. Rankings are fairly normally distributed except for Smug villagers. On the other hand, Cranky and Snooty both have a mean clearly above the overall mean.

# In[32]:


pd.pivot_table(combined_df, index = 'tier', values = 'Catchphrase', columns="Personality", aggfunc='count')


# ### 4. Style

# In[33]:


# generating value counts dataframe for each style type
style_ranking1 = combined_df.groupby('Style 1').mean(numeric_only=True)['overall_ranking'].to_frame().reset_index().sort_values('overall_ranking')
style_ranking2 = combined_df.groupby('Style 2').mean(numeric_only=True)['overall_ranking'].to_frame().reset_index().sort_values('overall_ranking')


# In[34]:


# combining the 2 style columns and finding a mean
style_ranking = style_ranking1.copy()
style_series = (style_ranking1['overall_ranking'] + style_ranking2['overall_ranking'])/2
style_ranking["overall_ranking"] = style_series


# In[35]:


style_ranking


# In[36]:


## -- STEFANOS: Disable plotting
# plt.figure(figsize=(20,5))
# sns.set(font_scale=1.4)
# plt.xticks(rotation=45)
# plt.axhline(overall_mean, color='r')
# sns.scatterplot(x='Style 1', y="overall_ranking", data=style_ranking, s=300)


# A very clear preference for Cute styled villagers. Simple Styled Villagers have a ranking mean just about equal to the overall mean, while other style villagers have a slightly above overall mean mean. 

# In[37]:


## -- STEFANOS: Disable plotting
# plt.figure(figsize=(7, 7))
# plt.axhline(overall_mean, color='r')
# sns.boxplot(x="Style 1", y='overall_ranking', data=combined_df)
# plt.title('Style 1')
# plt.figure(figsize=(7, 7))
# plt.axhline(overall_mean, color='r')
# sns.boxplot(x="Style 2", y='overall_ranking', data=combined_df)
# plt.title('Style 2')


# The clear preference is Cute style dressing, in both Style columns. In particular, in Style 2 column Cute Styled Villagers have a higher concetration in lower rankings. Other styles seem to have a fairly normally distributed ranking, with the exception of Active Style Villagers in Style 1, right skewed, but the ranking mean is significantly above the overall ranking mean.  

# In[38]:


pd.pivot_table(combined_df, index = 'tier', values = 'Catchphrase', columns="Style 1", aggfunc='count')


# In[39]:


pd.pivot_table(combined_df, index = 'tier', values = 'Catchphrase', columns="Style 2", aggfunc='count')


# # Conclusion
# We may come to the conclusion, that the following attributes contribute to a villager's popularity:
# - Gender: Despite Female Villagers having in general better popularity, this is likely due to the overwheling prescence of male villagers in the lowest tier. Other than the lowest tier, Male villagers in general perform slightly better.
# - Species: Octopus, Wolf, Deer and Cat villagers perform the best. 
# - Personality: Big Sister, Normal and Peppy villagers are in general the most popular. 
# - Style: Cute Style villagers are very clearly the most popular

# In[ ]:





# In[ ]:




