#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import os
# STEFANOS: Conditionally import Modin Pandas
import pandas as pd

# Input data files are available in the read-only "input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk(os.path.dirname('') + '/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


df = pd.read_table('input/imdb-official-movies-dataset/title-ratings.tsv', low_memory=False)
df2 = pd.read_table('input/imdb-official-movies-dataset/title-metadata.tsv',low_memory=False)
if "IREWR_LESS_REPLICATION" in os.environ and os.environ["IREWR_LESS_REPLICATION"] == "True":
    df2 = df2[:1_500_000]


# In[3]:


# Merging Rating and MetaData datsets into single dataframe w.r.t common column
main_df = pd.merge(df,df2,on='tconst')
main_df.head()


# **DATA Cleaning:**

# In[4]:


# creating a copy 
imdb_df = main_df.copy()


# **Currently, 'runtimeMinutes' column is in Minutes form thus we will convert it into Days:HH:MM:SS Format for better display of runtime information**

# In[5]:


imdb_df['runtimeMinutes'].value_counts()


# In[6]:


# Replace object type string '\N' with NaN First
imdb_df['runtime_delta'] = pd.to_numeric(imdb_df['runtimeMinutes'], errors='coerce', downcast="integer")


# In[7]:


# Using 'pd.to_timedelta' to convert minutes into desired format
imdb_df['runtime_delta'] = pd.to_timedelta(imdb_df['runtime_delta'], unit='m')


# **Similarly, The 'isAdult' column should contain only two values, '0: non-adult title, 1: adult title' but here As we can see dataFrame column consist of mixed data values like string, bool and int. Thus, Lets sort and restructure this column into a Bool data type.**

# In[8]:


imdb_df['isAdult'].unique()


# In[9]:


# Replace string and int by boolean
# we will leave out '2014' and '2020' as it is as there`s no way to confirm If they Adult or not not
imdb_df['isAdult'] = imdb_df['isAdult'].map({'0': 'Non_Adult_Title', '1': 'Adult_Title'})


# In[10]:


imdb_df.isAdult = imdb_df.isAdult.fillna('Unrated')
imdb_df['isAdult'].unique()


# **Creating a new column to provide information of show run timeline in better way by joinning 'startYear' and 'endYear' columns into one**

# In[11]:


# First, we should replace '\N' values to nan so next step
imdb_df['startYear'] = imdb_df['startYear'].replace('\\N', np.NaN)
imdb_df['endYear'] = imdb_df['endYear'].replace('\\N', np.NaN)


# In[12]:


# Now joinning them
# STEFANOS-DISABLE-FOR-MODIN
### ORIGINAL ###
# imdb_df['premiere_timeline'] = imdb_df[['startYear','endYear']].stack().groupby(level=0).agg('-'.join)
### COMPATIBLE WITH MODIN ###
# I tried a bunch of things, here's one:
#   imdb_df['premiere_timeline'] = imdb_df[['startYear','endYear']].stack().groupby(level=0).apply('-'.join)
# They don't seem to work. The problem seems to be with groupby(level=0). Generally, I'm not sure that Modin can
# groupby MultiIndex's.
# So, below I do the computation up to the stacking, just so that we can keep as much of it as possible (in case
# Modin, or some other system that uses the same benchmarks, can optimize it; we don't want to deprive this
# opportunity). Unfortunately, we cannot assign the result of stacking to premiere_timeline as they don't match.
# I just assign a dumb value to premiere_timeline so that it exists in the DF. It's not used later.

_ = imdb_df[['startYear','endYear']].stack()
imdb_df['premiere_timeline'] = 0


# **The data cleaning has been completed. However, Lets drop unneccessory columns from this dataframe for furthur analysis:**

# In[13]:


# Drop unneccessory columns from dataframe
columns_to_keep = ['tconst','averageRating','numVotes','titleType','primaryTitle',
                   'isAdult','premiere_timeline','runtime_delta']

imdb_df = imdb_df[columns_to_keep]

# Also renaming them with approriate name
renamed_cols = {'tconst':'IMDB_ID','averageRating':'Avg_Rating','numVotes':'Total_Votes',
                'titleType':'Title_Category','primaryTitle':'Title_Name','isAdult':'IN-18+'
                ,'premiere_timeline':'Air_time','runtime_delta':'Title_Runtime_Length'}

imdb_df.rename(columns = renamed_cols,inplace=True)
imdb_df.head()


# **Top 10 highest voted and rated Titles across all categories**

# In[14]:


top_10_highest = imdb_df.groupby(['Title_Name'])[
    ['Avg_Rating','Total_Votes']].max().sort_values(by=['Avg_Rating','Total_Votes'
                                                       ], ascending = False).reset_index().head(10)

top_10_highest


# **Many Many other titles received the Perfect Rating of 10 across all Category but My personal favorite movie of all time, 'Fight Club' tops the table with Perfect Rating and highest Individual votes received while 'Gladiator' and 'Die Hard' takes the 2nd and 3rd spot respectivly on IMDB table.**

# **Average votes and rating given to all categories**

# In[15]:


highest_rated_type = imdb_df.groupby(['Title_Category'])[['Avg_Rating','Total_Votes']].mean(
).sort_values(by=['Avg_Rating','Total_Votes'], ascending = False).reset_index().round(decimals = 1)

highest_rated_type


# **Seems like 'tvEpisode' Category has received the best avg. Rating of 7.4/title when compaired to all other title types but received very less avg. votes per title. On the other hand, 'Movie' Category tops the chart as they received highest 3532.7 votes/titles on IMDB.**

# **Category_wise longest running titles as per IMDB_database**

# In[16]:


longest_runtime = imdb_df.loc[imdb_df.groupby(['Title_Category'], 
                                              sort=False)['Title_Runtime_Length'].idxmax()
                             ][['Title_Category', 'Title_Name', 'Title_Runtime_Length']]

longest_runtime.sort_values(by='Title_Runtime_Length', ascending = False).reset_index(drop=True)


# **In our analysis, Movie named 'Logistics' has the longest run-time across all categories, at 35 Days and 17 Hours. While, 'The Longest Video on YouTube: 596.5 Hours' is in the second place.**

# **Now, Lets check Category_wise shortest running titles as per IMDB_database**

# In[17]:


shortest_runtime = imdb_df.loc[imdb_df.groupby(['Title_Category'], 
                                              sort=False)['Title_Runtime_Length'].idxmin()
                             ][['Title_Category', 'Title_Name', 'Title_Runtime_Length']]

shortest_runtime.reset_index(drop=True)


# **Short category film 'Awakening of Rip' holds the top place for being the shortest length title in IMDB_database.**

# **Now, Lets shed some light on Average Ratings of Non_Adult Rated Titles vs Adult Rated Titles**

# In[18]:


wanna_adult_or_not = imdb_df.groupby(['Title_Category','IN-18+']
                                    )['Avg_Rating'].agg('mean').round(1).unstack(fill_value= 0) 
wanna_adult_or_not.reset_index(inplace = True)
##### Visualise the chart in stack manner of bar type

# STEFANOS: Disable plotting
# plt.rcParams['figure.figsize'] = [15, 8]
# wanna_adult_or_not.plot(x='Title_Category', kind='bar', stacked=True, 
#                         title='Average Ratings of Non_Adult Rated Titles vs Adult Rated Titles')
# plt.xlabel('Title Categories')
# plt.ylabel('Average Ratings')
# plt.show()


# **'TvEpisode' category has received the highest Avg. Ratings in both Adult and Non-Adult rated title categories, However they there also the only title category among all whose some of the films/shows went Un-Rated.**

# **Here ends this project, where we dig deep into the dataframe. First, We cleaned it and then sorted the data structure making a display of meaningful data. If you want to give any suggestion and point out mistakes please feel free to contact. Thanks!**

# In[ ]:




