#!/usr/bin/env python
# coding: utf-8

# # Data Visualization
# 
# The purpose of this notebook is to practice data visualization, and hopefully communicate some best-practices along the way.
# 
# 
# # Please upvote if you find this useful
# 
# Some sources I'd reccomend for data visualization principles are:
# 
# - Storytelling with data, C. Knaflic
# 
# - The visual display of quantitative information, E. Tufte
# 
# - Better data visualizations, J. Schwabish
# 
# 
# I have other notebooks that also incoroprate some nice visuals:
# 
# **UK COVID-19 Vaccination Data Visualization**
# 
# https://www.kaggle.com/joshuaswords/uk-covid-19-vaccination-progress-data-vis
# 
# 
# **Exploratory Data Visualization - Student Performance**
# 
# https://www.kaggle.com/joshuaswords/data-visualisation-student-results
# 
# **HR Data Set - Visuals & Predictions**
# 
# https://www.kaggle.com/joshuaswords/awesome-hr-data-visualization-prediction
# 
# 
# **Visuals & Customer Segmentation**
# 
# https://www.kaggle.com/joshuaswords/data-visualization-clustering-mall-data
# 
# 
# As well as learning from other Kagglers who enjoy data viz.
# 
# With that said, let's go...

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
for dirname, _, filenames in os.walk('input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


# Importing libs
import numpy as np
# STEFANOS: Disable unneeded modules
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans, AffinityPropagation
# import matplotlib.pyplot as plt
# import seaborn as sns
# %matplotlib inline
import warnings
warnings.filterwarnings("ignore")
# import plotly as py
# import plotly.graph_objs as go
import os
# py.offline.init_notebook_mode(connected = True)
#print(os.listdir("input"))
import datetime as dt
# STEFANOS: Disable unneeded modules
# import missingno as msno
# plt.rcParams['figure.dpi'] = 140


# In[3]:


df = pd.read_csv('input/netflix-shows/netflix_titles.csv')

df.head(3)


# # -- STEFANOS -- Replicate Data

# In[4]:


factor = 500
df = pd.concat([df]*factor, ignore_index=True)


# In[5]:


# Missing data

for i in df.columns:
    null_rate = df[i].isna().sum() / len(df) * 100 
    if null_rate > 0 :
        print("{} null rate: {}%".format(i,round(null_rate,2)))


# - 5 columns have missing values, with Director missing 1/3 of the time

# # Dealing with the missing data
# 
# - This is always scenario dependant, but in this case, I will:
#     - replace blank countries with the mode (most common) country
#     - I want to keep director as it could be interesting to look at a certain director's films
#     - I want to keep cast as it could be interesting to look at a certain cast's films
#     

# In[6]:


# Replacments

df['country'] = df['country'].fillna(df['country'].mode()[0])


df['cast'].replace(np.nan, 'No Data',inplace  = True)
df['director'].replace(np.nan, 'No Data',inplace  = True)

# Drops

df.dropna(inplace=True)

# Drop Duplicates

# STEFANOS: Remove this because it makes the dataset same as the original.
# df.drop_duplicates(inplace= True)


# In[7]:


df.isnull().sum()


# In[8]:


# df.info()


# # Missing values dealt with, but the date isn't quite right yet...

# In[9]:


df["date_added"] = pd.to_datetime(df['date_added'])

df['month_added']=df['date_added'].dt.month
df['month_name_added']=df['date_added'].dt.month_name()
df['year_added'] = df['date_added'].dt.year

df.head(3)


# # Okay, let's visualize
# 
# # Where possible, I'll use the Netflix brand colours
# 
# https://brand.netflix.com/en/assets/brand-symbol/
# 
# 
# Using a consistent color palette is a great way to give your work credibility. It looks professional, and keeps the reader engaged. 
# 
# It's an easy-to-implement tip that really helps.

# In[10]:


# Palette
# STEFANOS: Disable plotting
# sns.palplot(['#221f1f', '#b20710', '#e50914','#f5f5f1'])

# plt.title("Netflix brand palette ",loc='left',fontfamily='serif',fontsize=15,y=1.2)
# plt.show()


# # Netflix through the years

# Netflix started as DVD rentals, and now they have an audience of over 150m people - this is their story...
# 
# Timeline code from Subin An's awesome notebook
# https://www.kaggle.com/subinium/awesome-visualization-with-titanic-dataset

# In[11]:


# STEFANOS: Disable plotting

# # Timeline code from Subin An's awesome notebook
# # https://www.kaggle.com/subinium/awesome-visualization-with-titanic-dataset


# from datetime import datetime

# ## these go on the numbers below
# tl_dates = [
#     "1997\nFounded",
#     "1998\nMail Service",
#     "2003\nGoes Public",
#     "2007\nStreaming service",
#     "2016\nGoes Global",
#     "2021\nNetflix & Chill"
# ]

# tl_x = [1, 2, 4, 5.3, 8,9]

# ## these go on the numbers
# tl_sub_x = [1.5,3,5,6.5,7]


# tl_sub_times = [
#     "1998","2000","2006","2010","2012"
# ]

# tl_text = [
#     "Netflix.com launched",
#     "Starts\nPersonal\nRecommendations","Billionth DVD Delivery","Canadian\nLaunch","UK Launch\n(my birthplace)"]



# # Set figure & Axes
# fig, ax = plt.subplots(figsize=(15, 4), constrained_layout=True)
# ax.set_ylim(-2, 1.75)
# ax.set_xlim(0, 10)


# # Timeline : line
# ax.axhline(0, xmin=0.1, xmax=0.9, c='#4a4a4a', zorder=1)


# # Timeline : Date Points
# ax.scatter(tl_x, np.zeros(len(tl_x)), s=120, c='#4a4a4a', zorder=2)
# ax.scatter(tl_x, np.zeros(len(tl_x)), s=30, c='#fafafa', zorder=3)
# # Timeline : Time Points
# ax.scatter(tl_sub_x, np.zeros(len(tl_sub_x)), s=50, c='#4a4a4a',zorder=4)

# # Date Text
# for x, date in zip(tl_x, tl_dates):
#     ax.text(x, -0.55, date, ha='center', 
#             fontfamily='serif', fontweight='bold',
#             color='#4a4a4a',fontsize=12)
    

# # Stemplot : vertical line
# levels = np.zeros(len(tl_sub_x))    
# levels[::2] = 0.3
# levels[1::2] = -0.3
# markerline, stemline, baseline = ax.stem(tl_sub_x, levels, use_line_collection=True)    
# plt.setp(baseline, zorder=0)
# plt.setp(markerline, marker=',', color='#4a4a4a')
# plt.setp(stemline, color='#4a4a4a')

# # Text
# for idx, x, time, txt in zip(range(1, len(tl_sub_x)+1), tl_sub_x, tl_sub_times, tl_text):
#     ax.text(x, 1.3*(idx%2)-0.5, time, ha='center', 
#             fontfamily='serif', fontweight='bold',
#             color='#4a4a4a' if idx!=len(tl_sub_x) else '#b20710', fontsize=11)
    
#     ax.text(x, 1.3*(idx%2)-0.6, txt, va='top', ha='center', 
#         fontfamily='serif',color='#4a4a4a' if idx!=len(tl_sub_x) else '#b20710')



# # Spine
# for spine in ["left", "top", "right", "bottom"]:
#     ax.spines[spine].set_visible(False)

# # Ticks    
# ax.set_xticks([]) 
# ax.set_yticks([]) 

# # Title
# ax.set_title("Netflix through the years", fontweight="bold", fontfamily='serif', fontsize=16, color='#4a4a4a')
# ax.text(2.4,1.57,"From DVD rentals to a global audience of over 150m people - is it time for Netflix to Chill?", fontfamily='serif', fontsize=12, color='#4a4a4a')

# plt.show()


# # Content - Let's explore
# 
# Now we've seen how Netflix came to dominate our TV screens, let's have a look at the content they offer...

# In[12]:


# For viz: Ratio of Movies & TV shows

x=df.groupby(['type'])['type'].count()
y=len(df)
r=((x/y)).round(2)

mf_ratio = pd.DataFrame(r).T


# In[13]:


# STEFANOS: Disable plotting


# fig, ax = plt.subplots(1,1,figsize=(6.5, 2.5))

# ax.barh(mf_ratio.index, mf_ratio['Movie'], 
#         color='#b20710', alpha=0.9, label='Male')
# ax.barh(mf_ratio.index, mf_ratio['TV Show'], left=mf_ratio['Movie'], 
#         color='#221f1f', alpha=0.9, label='Female')

# ax.set_xlim(0, 1)
# ax.set_xticks([])
# ax.set_yticks([])
# #ax.set_yticklabels(mf_ratio.index, fontfamily='serif', fontsize=11)


# # movie percentage
# for i in mf_ratio.index:
#     ax.annotate(f"{int(mf_ratio['Movie'][i]*100)}%", 
#                    xy=(mf_ratio['Movie'][i]/2, i),
#                    va = 'center', ha='center',fontsize=40, fontweight='light', fontfamily='serif',
#                    color='white')

#     ax.annotate("Movie", 
#                    xy=(mf_ratio['Movie'][i]/2, -0.25),
#                    va = 'center', ha='center',fontsize=15, fontweight='light', fontfamily='serif',
#                    color='white')
    
    
# for i in mf_ratio.index:
#     ax.annotate(f"{int(mf_ratio['TV Show'][i]*100)}%", 
#                    xy=(mf_ratio['Movie'][i]+mf_ratio['TV Show'][i]/2, i),
#                    va = 'center', ha='center',fontsize=40, fontweight='light', fontfamily='serif',
#                    color='white')
#     ax.annotate("TV Show", 
#                    xy=(mf_ratio['Movie'][i]+mf_ratio['TV Show'][i]/2, -0.25),
#                    va = 'center', ha='center',fontsize=15, fontweight='light', fontfamily='serif',
#                    color='white')






# # Title & Subtitle
# fig.text(0.125,1.03,'Movie & TV Show distribution', fontfamily='serif',fontsize=15, fontweight='bold')
# fig.text(0.125,0.92,'We see vastly more movies than TV shows on Netflix.',fontfamily='serif',fontsize=12)  

# for s in ['top', 'left', 'right', 'bottom']:
#     ax.spines[s].set_visible(False)
    


# #ax.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.06))

# # Removing legend due to labelled plot
# ax.legend().set_visible(False)
# plt.show()









for i in mf_ratio.index:
    _ = int(mf_ratio['Movie'][i]*100)
    _ = mf_ratio['Movie'][i]/2
    _ = mf_ratio['Movie'][i]/2
    
for i in mf_ratio.index:
    _ = int(mf_ratio['TV Show'][i]*100)
    _ = mf_ratio['Movie'][i]+mf_ratio['TV Show'][i]/2
    _ = mf_ratio['Movie'][i]+mf_ratio['TV Show'][i]/2


# # By Country
# 
# So we now know there are much more movies than TV shows on Netflix (which surprises me!).
# 
# What about if we look at content by country? 
# 
# I would imagine that the USA will have the most content. I wonder how my country, the UK, will compare?

# In[14]:


# Quick feature engineering

# Helper column for various plots
df['count'] = 1

# Many productions have several countries listed - this will skew our results , we'll grab the first one mentioned

# Lets retrieve just the first country
df['first_country'] = df['country'].apply(lambda x: x.split(",")[0])
df['first_country'].head()

# Rating ages from this notebook: https://www.kaggle.com/andreshg/eda-beginner-to-expert-plotly (thank you!)

ratings_ages = {
    'TV-PG': 'Older Kids',
    'TV-MA': 'Adults',
    'TV-Y7-FV': 'Older Kids',
    'TV-Y7': 'Older Kids',
    'TV-14': 'Teens',
    'R': 'Adults',
    'TV-Y': 'Kids',
    'NR': 'Adults',
    'PG-13': 'Teens',
    'TV-G': 'Kids',
    'PG': 'Older Kids',
    'G': 'Kids',
    'UR': 'Adults',
    'NC-17': 'Adults'
}

df['target_ages'] = df['rating'].replace(ratings_ages)
df['target_ages'].unique()

# Genre

df['genre'] = df['listed_in'].apply(lambda x :  x.replace(' ,',',').replace(', ',',').split(',')) 

# Reducing name length

df['first_country'].replace('United States', 'USA', inplace=True)
df['first_country'].replace('United Kingdom', 'UK',inplace=True)
df['first_country'].replace('South Korea', 'S. Korea',inplace=True)


# In[15]:


data = df.groupby('first_country')['count'].sum().sort_values(ascending=False)[:10]

# STEFANOS: Disable plotting

# # Plot

# color_map = ['#f5f5f1' for _ in range(10)]
# color_map[0] = color_map[1] = color_map[2] =  '#b20710' # color highlight

# fig, ax = plt.subplots(1,1, figsize=(12, 6))
# ax.bar(data.index, data, width=0.5, 
#        edgecolor='darkgray',
#        linewidth=0.6,color=color_map)

# #annotations
# for i in data.index:
#     ax.annotate(f"{data[i]}", 
#                    xy=(i, data[i] + 150), #i like to change this to roughly 5% of the highest cat
#                    va = 'center', ha='center',fontweight='light', fontfamily='serif')



# # Remove border from plot

# for s in ['top', 'left', 'right']:
#     ax.spines[s].set_visible(False)
    
# # Tick labels

# ax.set_xticklabels(data.index, fontfamily='serif', rotation=0)

# # Title and sub-title

# fig.text(0.09, 1, 'Top 10 countries on Netflix', fontsize=15, fontweight='bold', fontfamily='serif')
# fig.text(0.09, 0.95, 'The three most frequent countries have been highlighted.', fontsize=12, fontweight='light', fontfamily='serif')

# fig.text(1.1, 1.01, 'Insight', fontsize=15, fontweight='bold', fontfamily='serif')

# fig.text(1.1, 0.67, '''
# The most prolific producers of
# content for Netflix are, primarily,
# the USA, with India and the UK
# a significant distance behind.

# It makes sense that the USA produces 
# the most content as, afterall, 
# Netflix is a US company.
# '''
#          , fontsize=12, fontweight='light', fontfamily='serif')

# ax.grid(axis='y', linestyle='-', alpha=0.4)   

# grid_y_ticks = np.arange(0, 4000, 500) # y ticks, min, max, then step
# ax.set_yticks(grid_y_ticks)
# ax.set_axisbelow(True)

# #Axis labels

# #plt.xlabel("Country", fontsize=12, fontweight='light', fontfamily='serif',loc='left',y=-1.5)
# #plt.ylabel("Count", fontsize=12, fontweight='light', fontfamily='serif')
#  #plt.legend(loc='upper right')
    
# # thicken the bottom line if you want to
# plt.axhline(y = 0, color = 'black', linewidth = 1.3, alpha = .7)

# ax.tick_params(axis='both', which='major', labelsize=12)


# import matplotlib.lines as lines
# l1 = lines.Line2D([1, 1], [0, 1], transform=fig.transFigure, figure=fig,color='black',lw=0.2)
# fig.lines.extend([l1])

# ax.tick_params(axis=u'both', which=u'both',length=0)

# plt.show()


# As predicted, the USA dominates. 
# 
# The UK is a top contender too, but still some way behind India.
# 
# How does content by country vary? 

# In[16]:


country_order = df['first_country'].value_counts()[:11].index
### STEFANOS: I didn't find a way to run the original with Modin. We're converting to Pandas and then
### back to Modin.
### ORIGINAL:
# data_q2q3 = df[['type', 'first_country']].groupby('first_country')['type'].value_counts().unstack().loc[country_order]
### MODIFIED:
if "IREWR_WITH_MODIN" in os.environ and os.environ["IREWR_WITH_MODIN"] == "True":
  df = df._to_pandas()

data_q2q3 = df[['type', 'first_country']].groupby('first_country')['type'].value_counts().unstack().loc[country_order]
if "IREWR_WITH_MODIN" in os.environ and os.environ["IREWR_WITH_MODIN"] == "True":
  df = pd.DataFrame(df)
  data_q2q3 = pd.DataFrame(data_q2q3)

data_q2q3['sum'] = data_q2q3.sum(axis=1)
data_q2q3_ratio = (data_q2q3.T / data_q2q3['sum']).T[['Movie', 'TV Show']].sort_values(by='Movie',ascending=False)[::-1]


# STEFANOS: Disable plotting

# ###
# fig, ax = plt.subplots(1,1,figsize=(15, 8),)

# ax.barh(data_q2q3_ratio.index, data_q2q3_ratio['Movie'], 
#         color='#b20710', alpha=0.8, label='Movie')
# ax.barh(data_q2q3_ratio.index, data_q2q3_ratio['TV Show'], left=data_q2q3_ratio['Movie'], 
#         color='#221f1f', alpha=0.8, label='TV Show')


# ax.set_xlim(0, 1)
# ax.set_xticks([])
# ax.set_yticklabels(data_q2q3_ratio.index, fontfamily='serif', fontsize=11)

# # male percentage
# for i in data_q2q3_ratio.index:
#     ax.annotate(f"{data_q2q3_ratio['Movie'][i]*100:.3}%", 
#                    xy=(data_q2q3_ratio['Movie'][i]/2, i),
#                    va = 'center', ha='center',fontsize=12, fontweight='light', fontfamily='serif',
#                    color='white')

# for i in data_q2q3_ratio.index:
#     ax.annotate(f"{data_q2q3_ratio['TV Show'][i]*100:.3}%", 
#                    xy=(data_q2q3_ratio['Movie'][i]+data_q2q3_ratio['TV Show'][i]/2, i),
#                    va = 'center', ha='center',fontsize=12, fontweight='light', fontfamily='serif',
#                    color='white')
    

# fig.text(0.13, 0.93, 'Top 10 countries Movie & TV Show split', fontsize=15, fontweight='bold', fontfamily='serif')   
# fig.text(0.131, 0.89, 'Percent Stacked Bar Chart', fontsize=12,fontfamily='serif')   

# for s in ['top', 'left', 'right', 'bottom']:
#     ax.spines[s].set_visible(False)
    
# #ax.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.06))

# fig.text(0.75,0.9,"Movie", fontweight="bold", fontfamily='serif', fontsize=15, color='#b20710')
# fig.text(0.81,0.9,"|", fontweight="bold", fontfamily='serif', fontsize=15, color='black')
# fig.text(0.82,0.9,"TV Show", fontweight="bold", fontfamily='serif', fontsize=15, color='#221f1f')


# fig.text(1.1, 0.93, 'Insight', fontsize=15, fontweight='bold', fontfamily='serif')

# fig.text(1.1, 0.44, '''
# Interestingly, Netflix in India
# is made up nearly entirely of Movies. 

# Bollywood is big business, and perhaps
# the main focus of this industry is Movies
# and not TV Shows.

# South Korean Netflix on the other hand is 
# almost entirely TV Shows.

# The underlying resons for the difference 
# in content must be due to market research
# conducted by Netflix.
# '''
#          , fontsize=12, fontweight='light', fontfamily='serif')



# import matplotlib.lines as lines
# l1 = lines.Line2D([1, 1], [0, 1], transform=fig.transFigure, figure=fig,color='black',lw=0.2)
# fig.lines.extend([l1])




# ax.tick_params(axis='both', which='major', labelsize=12)
# ax.tick_params(axis=u'both', which=u'both',length=0)

# plt.show()


# As I've noted in the insights on the plot, it is really interesting to see how the split of TV Shows and Movies varies by country.
# 
# South Korea is dominated by TV Shows - why is this? I am a huge fan of South Korean cinema so I know they have a great movie selection.
# 
# Equally, India is dominated by Movies. I think this might be due to Bollywood - comment below if you have any other ideas!
# 
# # Ratings
# 
# Let's briefly check out how ratings are distributed

# In[17]:


order = pd.DataFrame(df.groupby('rating')['count'].sum().sort_values(ascending=False).reset_index())
rating_order = list(order['rating'])


# In[18]:


### STEFANOS: I didn't find a way to run the original with Modin. We're converting to Pandas and then
### back to Modin.
### ORIGINAL:
# mf = df.groupby('type')['rating'].value_counts().unstack().sort_index().fillna(0).astype(int)[rating_order]
### MODIFIED:

if "IREWR_WITH_MODIN" in os.environ and os.environ["IREWR_WITH_MODIN"] == "True":
  df = df._to_pandas()

mf = df.groupby('type')['rating'].value_counts().unstack().sort_index().fillna(0).astype(int)[rating_order]
if "IREWR_WITH_MODIN" in os.environ and os.environ["IREWR_WITH_MODIN"] == "True":
  df = pd.DataFrame(df)
  mf = pd.DataFrame(mf)

movie = mf.loc['Movie']
tv = - mf.loc['TV Show']

# STEFANOS: Disable plotting

# fig, ax = plt.subplots(1,1, figsize=(12, 6))
# ax.bar(movie.index, movie, width=0.5, color='#b20710', alpha=0.8, label='Movie')
# ax.bar(tv.index, tv, width=0.5, color='#221f1f', alpha=0.8, label='TV Show')
# #ax.set_ylim(-35, 50)

# # Annotations
# for i in tv.index:
#     ax.annotate(f"{-tv[i]}", 
#                    xy=(i, tv[i] - 60),
#                    va = 'center', ha='center',fontweight='light', fontfamily='serif',
#                    color='#4a4a4a')   

# for i in movie.index:
#     ax.annotate(f"{movie[i]}", 
#                    xy=(i, movie[i] + 60),
#                    va = 'center', ha='center',fontweight='light', fontfamily='serif',
#                    color='#4a4a4a')
    
 

# for s in ['top', 'left', 'right', 'bottom']:
#     ax.spines[s].set_visible(False)

# ax.set_xticklabels(mf.columns, fontfamily='serif')
# ax.set_yticks([])    

# ax.legend().set_visible(False)
# fig.text(0.16, 1, 'Rating distribution by Film & TV Show', fontsize=15, fontweight='bold', fontfamily='serif')
# fig.text(0.16, 0.89, 
# '''We observe that some ratings are only applicable to Movies. 
# The most common for both Movies & TV Shows are TV-MA and TV-14.
# '''

# , fontsize=12, fontweight='light', fontfamily='serif')


# fig.text(0.755,0.924,"Movie", fontweight="bold", fontfamily='serif', fontsize=15, color='#b20710')
# fig.text(0.815,0.924,"|", fontweight="bold", fontfamily='serif', fontsize=15, color='black')
# fig.text(0.825,0.924,"TV Show", fontweight="bold", fontfamily='serif', fontsize=15, color='#221f1f')

# plt.show()


# # How has content been added over the years?

# As we saw in the timeline at the start of this analysis, Netflix went global in 2016 - and it is extremely noticeable in this plot.
# 
# The increase is Movie content is remarkable.

# In[19]:


# STEFANOS: Disable plotting

# fig, ax = plt.subplots(1, 1, figsize=(12, 6))
# color = ["#b20710", "#221f1f"]

# for i, mtv in enumerate(df['type'].value_counts().index):
#     mtv_rel = df[df['type']==mtv]['year_added'].value_counts().sort_index()
#     ax.plot(mtv_rel.index, mtv_rel, color=color[i], label=mtv)
#     ax.fill_between(mtv_rel.index, 0, mtv_rel, color=color[i], alpha=0.9)
    
# ax.yaxis.tick_right()
    
# ax.axhline(y = 0, color = 'black', linewidth = 1.3, alpha = .7)

# #ax.set_ylim(0, 50)
# #ax.legend(loc='upper left')
# for s in ['top', 'right','bottom','left']:
#     ax.spines[s].set_visible(False)

# ax.grid(False)

# ax.set_xlim(2008,2020)
# plt.xticks(np.arange(2008, 2021, 1))

# fig.text(0.13, 0.85, 'Movies & TV Shows added over time', fontsize=15, fontweight='bold', fontfamily='serif')
# fig.text(0.13, 0.59, 
# '''We see a slow start for Netflix over several years. 
# Things begin to pick up in 2015 and then there is a 
# rapid increase from 2016.

# It looks like content additions have slowed down in 2020, 
# likely due to the COVID-19 pandemic.
# '''

# , fontsize=12, fontweight='light', fontfamily='serif')


# fig.text(0.13,0.2,"Movie", fontweight="bold", fontfamily='serif', fontsize=15, color='#b20710')
# fig.text(0.19,0.2,"|", fontweight="bold", fontfamily='serif', fontsize=15, color='black')
# fig.text(0.2,0.2,"TV Show", fontweight="bold", fontfamily='serif', fontsize=15, color='#221f1f')

# ax.tick_params(axis=u'both', which=u'both',length=0)

# plt.show()


# # We can view the same plot, but as a cumulative total...

# In[20]:


### STEFANOS: I didn't find a way to run the original with Modin. We're converting to Pandas and then
### back to Modin.
### ORIGINAL:
# data_sub = df.groupby('type')['year_added'].value_counts().unstack().fillna(0).loc[['TV Show','Movie']].cumsum(axis=0).T
### MODIFIED:

if "IREWR_WITH_MODIN" in os.environ and os.environ["IREWR_WITH_MODIN"] == "True":
  df = df._to_pandas()

data_sub = df.groupby('type')['year_added'].value_counts().unstack().fillna(0).loc[['TV Show','Movie']].cumsum(axis=0).T
if "IREWR_WITH_MODIN" in os.environ and os.environ["IREWR_WITH_MODIN"] == "True":
  df = pd.DataFrame(df)
  data_sub = pd.DataFrame(data_sub)

# STEFANOS: Disable plotting

# fig, ax = plt.subplots(1, 1, figsize=(12, 6))
# color = ["#b20710", "#221f1f"]

# for i, mtv in enumerate(df['type'].value_counts().index):
#     mtv_rel = data_sub[mtv]
#     ax.fill_between(mtv_rel.index, 0, mtv_rel, color=color[i], label=mtv,alpha=0.9)
    

    
# ax.yaxis.tick_right()
    
# ax.axhline(y = 0, color = 'black', linewidth = 1.3, alpha = .7)

# #ax.set_ylim(0, 50)
# #ax.legend(loc='upper left')
# for s in ['top', 'right','bottom','left']:
#     ax.spines[s].set_visible(False)

# ax.grid(False)

# ax.set_xlim(2008,2020)
# plt.xticks(np.arange(2008, 2021, 1))

# fig.text(0.13, 0.85, 'Movies & TV Shows added over time [Cumulative Total]', fontsize=15, fontweight='bold', fontfamily='serif')
# fig.text(0.13, 0.58, 
# '''Netflix peak global content amount was in 2019.

# It appears that Netflix has focused more attention
# on increasing Movie content that TV Shows. 
# Movies have increased much more dramatically
# than TV shows.
# '''

# , fontsize=12, fontweight='light', fontfamily='serif')



# fig.text(0.13,0.2,"Movie", fontweight="bold", fontfamily='serif', fontsize=15, color='#b20710')
# fig.text(0.19,0.2,"|", fontweight="bold", fontfamily='serif', fontsize=15, color='black')
# fig.text(0.2,0.2,"TV Show", fontweight="bold", fontfamily='serif', fontsize=15, color='#221f1f')

# ax.tick_params(axis=u'both', which=u'both',length=0)


# plt.show()


# # Month-by-Month
# 
# We've seen how content has increased over the years, but are there certain months that, on average, tend to enjoy more content being added?
# 
# I'll show this in a couple of ways - a cumulative year view, and also as a radial plot...

# In[21]:


month_order = ['January',
 'February',
 'March',
 'April',
 'May',
 'June',
 'July',
 'August',
 'September',
 'October',
 'November',
 'December']

df['month_name_added'] = pd.Categorical(df['month_name_added'], categories=month_order, ordered=True)


# In[22]:


### STEFANOS: I didn't find a way to run the original with Modin. We're converting to Pandas and then
### back to Modin.
### ORIGINAL:
# data_sub = df.groupby('type')['month_name_added'].value_counts().unstack().fillna(0).loc[['TV Show','Movie']].cumsum(axis=0).T
### MODIFIED:

if "IREWR_WITH_MODIN" in os.environ and os.environ["IREWR_WITH_MODIN"] == "True":
  df = df._to_pandas()

data_sub = df.groupby('type')['month_name_added'].value_counts().unstack().fillna(0).loc[['TV Show','Movie']].cumsum(axis=0).T
if "IREWR_WITH_MODIN" in os.environ and os.environ["IREWR_WITH_MODIN"] == "True":
  df = pd.DataFrame(df)
  data_sub = pd.DataFrame(data_sub)

# STEFANOS: Disable plotting

# fig, ax = plt.subplots(1, 1, figsize=(12, 6))
# color = ["#b20710", "#221f1f"]

# for i, mtv in enumerate(df['type'].value_counts().index):
#     mtv_rel = data_sub[mtv]
#     ax.fill_between(mtv_rel.index, 0, mtv_rel, color=color[i], label=mtv,alpha=0.9)
    

    
# ax.yaxis.tick_right()
    
# ax.axhline(y = 0, color = 'black', linewidth = 1.3, alpha = .4)

# #ax.set_ylim(0, 50)
# #ax.legend(loc='upper left')
# for s in ['top', 'right','bottom','left']:
#     ax.spines[s].set_visible(False)

# ax.grid(False)
# ax.set_xticklabels(data_sub.index, fontfamily='serif', rotation=0)
# ax.margins(x=0) # remove white spaces next to margins

# #ax.set_xlim(2008,2020)
# #plt.xticks(np.arange(2008, 2021, 1))

# fig.text(0.13, 0.95, 'Content added by month [Cumulative Total]', fontsize=15, fontweight='bold', fontfamily='serif')
# fig.text(0.13, 0.905, 
# "The end & beginnings of each year seem to be Netflix's preference for adding content."

# , fontsize=12, fontweight='light', fontfamily='serif')



# fig.text(0.13,0.855,"Movie", fontweight="bold", fontfamily='serif', fontsize=15, color='#b20710')
# fig.text(0.19,0.855,"|", fontweight="bold", fontfamily='serif', fontsize=15, color='black')
# fig.text(0.2,0.855,"TV Show", fontweight="bold", fontfamily='serif', fontsize=15, color='#221f1f')


# ax.tick_params(axis=u'both', which=u'both',length=0)

# plt.show()


# # What about a more interesting way to view how content is added across the year?
# 
# Sometimes visualizations should be eye-catching & attention grabbing - I think this visual acheives that, even if it isn't the most precise.
# 
# By highlighting certain months, the reader's eye is drawn exactly where we want it. 

# In[23]:


data_sub2 = data_sub

data_sub2['Value'] = data_sub2['Movie'] + data_sub2['TV Show']
data_sub2 = data_sub2.reset_index()

df_polar = data_sub2.sort_values(by='month_name_added',ascending=False)


color_map = ['#221f1f' for _ in range(12)]
color_map[0] = color_map[11] =  '#b20710' # color highlight


# initialize the figure
# STEFANOS: Disable plotting
# plt.figure(figsize=(8,8))
# ax = plt.subplot(111, polar=True)
# plt.axis('off')

# Constants = parameters controling the plot layout:
upperLimit = 30
lowerLimit = 1
labelPadding = 30

# Compute max and min in the dataset
max = df_polar['Value'].max()

# Let's compute heights: they are a conversion of each item value in those new coordinates
# In our example, 0 in the dataset will be converted to the lowerLimit (10)
# The maximum will be converted to the upperLimit (100)
slope = (max - lowerLimit) / max
heights = slope * df_polar.Value + lowerLimit

# Compute the width of each bar. In total we have 2*Pi = 360°
width = 2*np.pi / len(df_polar.index)

# Compute the angle each bar is centered on:
indexes = list(range(1, len(df_polar.index)+1))
angles = [element * width for element in indexes]
angles

# STEFANOS: Disable plotting

# # Draw bars
# bars = ax.bar(
#     x=angles, 
#     height=heights, 
#     width=width, 
#     bottom=lowerLimit,
#     linewidth=2, 
#     edgecolor="white",
#     color=color_map,alpha=0.8
# )

# # Add labels
# for bar, angle, height, label in zip(bars,angles, heights, df_polar["month_name_added"]):

#     # Labels are rotated. Rotation must be specified in degrees :(
#     rotation = np.rad2deg(angle)

#     # Flip some labels upside down
#     alignment = ""
#     if angle >= np.pi/2 and angle < 3*np.pi/2:
#         alignment = "right"
#         rotation = rotation + 180
#     else: 
#         alignment = "left"

#     # Finally add the labels
#     ax.text(
#         x=angle, 
#         y=lowerLimit + bar.get_height() + labelPadding, 
#         s=label, 
#         ha=alignment, fontsize=10,fontfamily='serif',
#         va='center', 
#         rotation=rotation, 
#         rotation_mode="anchor") 


# Yes, December & January are definitely the best months for new content. Maybe Netflix knows that people have a lot of time off from work over this period and that it is a good time to reel people in?
# 
# February is the worst - why might this be? Ideas welcomed!

# # Movie Genres
# 
# Let's now explore movie genres a little...

# In[24]:


# Genres
from sklearn.preprocessing import MultiLabelBinarizer 

import matplotlib.colors


# Custom colour map based on Netflix palette
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ['#221f1f', '#b20710','#f5f5f1'])



def genre_heatmap(df, title):
    df['genre'] = df['listed_in'].apply(lambda x :  x.replace(' ,',',').replace(', ',',').split(',')) 
    Types = []
    for i in df['genre']: Types += i
    Types = set(Types)
    print("There are {} types in the Netflix {} Dataset".format(len(Types),title))    
    test = df['genre']
# STEFANOS: Disable ML and plotting
#     mlb = MultiLabelBinarizer()
#     res = pd.DataFrame(mlb.fit_transform(test), columns=mlb.classes_, index=test.index)
#     corr = res.corr()
#     mask = np.zeros_like(corr, dtype=np.bool)
#     mask[np.triu_indices_from(mask)] = True
#     fig, ax = plt.subplots(figsize=(10, 7))
#     fig.text(.54,.88,'Genre correlation', fontfamily='serif',fontweight='bold',fontsize=15)
#     fig.text(.75,.665,
#             '''
#              It is interesting that Independant Movies
#              tend to be Dramas. 
             
#              Another observation is that 
#              Internatinal Movies are rarely
#              in the Children's genre.
#              ''', fontfamily='serif',fontsize=12,ha='right')
#     pl = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, vmin=-.3, center=0, square=True, linewidths=2.5)
    
#     plt.show()


# In[25]:


df_tv = df[df["type"] == "TV Show"]
df_movies = df[df["type"] == "Movie"]


genre_heatmap(df_movies, 'Movie')
# STEFANOS: Disable plotting
# plt.show()


# In[26]:


data = df.groupby('first_country')[['first_country','count']].sum().sort_values(by='count',ascending=False).reset_index()[:10]
data = data['first_country']


df_heatmap = df.loc[df['first_country'].isin(data)]


# In[27]:


df_heatmap = pd.crosstab(df_heatmap['first_country'],df_heatmap['target_ages'],normalize = "index").T


# # Target Ages
# 
# Does Netflix uniformly target certain demographics? Or does this vary by country?
# 
# 

# In[28]:


# STEFANOS: Disable plotting

# fig, ax = plt.subplots(1, 1, figsize=(12, 12))

# country_order2 = ['USA', 'India', 'UK', 'Canada', 'Japan', 'France', 'S. Korea', 'Spain',
#        'Mexico', 'Turkey']

# age_order = ['Kids','Older Kids','Teens','Adults']

# sns.heatmap(df_heatmap.loc[age_order,country_order2],cmap=cmap,square=True, linewidth=2.5,cbar=False,
#             annot=True,fmt='1.0%',vmax=.6,vmin=0.05,ax=ax,annot_kws={"fontsize":12})

# ax.spines['top'].set_visible(True)


# fig.text(.99, .725, 'Target ages proportion of total content by country', fontweight='bold', fontfamily='serif', fontsize=15,ha='right')   
# fig.text(0.99, 0.7, 'Here we see interesting differences between countries. Most shows in India are targeted to teens, for instance.',ha='right', fontsize=12,fontfamily='serif') 

# ax.set_yticklabels(ax.get_yticklabels(), fontfamily='serif', rotation = 0, fontsize=11)
# ax.set_xticklabels(ax.get_xticklabels(), fontfamily='serif', rotation=90, fontsize=11)

# ax.set_ylabel('')    
# ax.set_xlabel('')
# ax.tick_params(axis=u'both', which=u'both',length=0)
# plt.tight_layout()
# plt.show()


# Very interesting results. 
# 
# It is also interesting to note similarities between culturally similar countries - the US & UK are closey aligned with their Netflix target ages, yet vastly different to say, India or Japan!

# # Let's have a quick look at the lag between when content is released and when it is added on Netflix
# 
# Spain looks to have a lot of new content. Great for them!

# In[29]:


# Data

df_movies
df_tv

### Relevant groupings

data = df_movies.groupby('first_country')[['first_country','count']].sum().sort_values(by='count',ascending=False).reset_index()[:10]
data = data['first_country']
df_loli = df_movies.loc[df_movies['first_country'].isin(data)]

loli = df_loli.groupby('first_country')['release_year','year_added'].mean().round()


# Reorder it following the values of the first value
ordered_df = loli.sort_values(by='release_year')

ordered_df_rev = loli.sort_values(by='release_year',ascending=False)

# STEFANOS: Disable plotting
# my_range=range(1,len(loli.index)+1)


# fig, ax = plt.subplots(1, 1, figsize=(7, 5))

# fig.text(0.13, 0.9, 'How old are the movies? [Average]', fontsize=15, fontweight='bold', fontfamily='serif')
# plt.hlines(y=my_range, xmin=ordered_df['release_year'], xmax=ordered_df['year_added'], color='grey', alpha=0.4)
# plt.scatter(ordered_df['release_year'], my_range, color='#221f1f',s=100, alpha=0.9, label='Average release date')
# plt.scatter(ordered_df['year_added'], my_range, color='#b20710',s=100, alpha=0.9 , label='Average added date')
# #plt.legend()

# for s in ['top', 'left', 'right', 'bottom']:
#     ax.spines[s].set_visible(False)
    

# # Removes the tick marks but keeps the labels
# ax.tick_params(axis=u'both', which=u'both',length=0)
# # Move Y axis to the right side
# ax.yaxis.tick_right()

# plt.yticks(my_range, ordered_df.index)
# plt.yticks(fontname = "serif",fontsize=12)

# # Custome legend
# fig.text(0.19,0.175,"Released", fontweight="bold", fontfamily='serif', fontsize=12, color='#221f1f')
# fig.text(0.76,0.175,"Added", fontweight="bold", fontfamily='serif', fontsize=12, color='#b20710')


# fig.text(0.13, 0.46, 
# '''The average gap between when 
# content is released, and when it
# is then added on Netflix varies
# by country. 

# In Spain, Netflix appears to be 
# dominated by newer movies 
# whereas Egypt & India have
# an older average movie.
# '''

# , fontsize=12, fontweight='light', fontfamily='serif')


# #plt.xlabel('Year')
# #plt.ylabel('Country')
# plt.show()


# What about TV shows...

# In[30]:


data = df_tv.groupby('first_country')[['first_country','count']].sum().sort_values(by='count',ascending=False).reset_index()[:10]
data = data['first_country']
df_loli = df_tv.loc[df_tv['first_country'].isin(data)]

loli = df_loli.groupby('first_country')['release_year','year_added'].mean().round()


# Reorder it following the values of the first value:
ordered_df = loli.sort_values(by='release_year')

ordered_df_rev = loli.sort_values(by='release_year',ascending=False)

# STEFANOS: Disable plotting


# my_range=range(1,len(loli.index)+1)


# fig, ax = plt.subplots(1, 1, figsize=(7, 5))

# fig.text(0.13, 0.9, 'How old are the TV shows? [Average]', fontsize=15, fontweight='bold', fontfamily='serif')
# plt.hlines(y=my_range, xmin=ordered_df['release_year'], xmax=ordered_df['year_added'], color='grey', alpha=0.4)
# plt.scatter(ordered_df['release_year'], my_range, color='#221f1f',s=100, alpha=0.9, label='Average release date')
# plt.scatter(ordered_df['year_added'], my_range, color='#b20710',s=100, alpha=0.9 , label='Average added date')
# #plt.legend()

# for s in ['top', 'left', 'right', 'bottom']:
#     ax.spines[s].set_visible(False)
    
# ax.yaxis.tick_right()
# plt.yticks(my_range, ordered_df.index)
# plt.yticks(fontname = "serif",fontsize=12)


# fig.text(0.19,0.175,"Released", fontweight="bold", fontfamily='serif', fontsize=12, color='#221f1f')

# fig.text(0.47,0.175,"Added", fontweight="bold", fontfamily='serif', fontsize=12, color='#b20710')


# fig.text(0.13, 0.42, 
# '''The gap for TV shows seems
# more regular than for movies.

# This is likely due to subsequent
# series being released
# year-on-year.

# Spain seems to have
# the newest content
# overall.
# '''

# , fontsize=12, fontweight='light', fontfamily='serif')


# ax.tick_params(axis=u'both', which=u'both',length=0)
# #plt.xlabel('Value of the variables')
# #plt.ylabel('Group')
# plt.show()


# In[31]:


us_ind = df[(df['first_country'] == 'USA') | (df['first_country'] == 'India' )]


### STEFANOS: I didn't find a way to run the original with Modin. We're converting to Pandas and then
### back to Modin.
### ORIGINAL:
# data_sub = df.groupby('first_country')['year_added'].value_counts().unstack().fillna(0).loc[['USA','India']].cumsum(axis=0).T
### MODIFIED:

if "IREWR_WITH_MODIN" in os.environ and os.environ["IREWR_WITH_MODIN"] == "True":
  df = df._to_pandas()

data_sub = df.groupby('first_country')['year_added'].value_counts().unstack().fillna(0).loc[['USA','India']].cumsum(axis=0).T
if "IREWR_WITH_MODIN" in os.environ and os.environ["IREWR_WITH_MODIN"] == "True":
  df = pd.DataFrame(df)
  data_sub = pd.DataFrame(data_sub)


# # USA & India
# 
# As the two largest content countries, it might be fun to compare the two

# In[32]:


# STEFANOS: Disable plotting

# fig, ax = plt.subplots(1, 1, figsize=(12, 6))
# color = ['#221f1f', '#b20710','#f5f5f1']

# for i, hs in enumerate(us_ind['first_country'].value_counts().index):
#     hs_built = us_ind[us_ind['first_country']==hs]['year_added'].value_counts().sort_index()
#     ax.plot(hs_built.index, hs_built, color=color[i], label=hs)
#     #ax.fill_between(hs_built.index, 0, hs_built, color=color[i], alpha=0.4)
#     ax.fill_between(hs_built.index, 0, hs_built, color=color[i], label=hs)
    

# ax.set_ylim(0, 1000)
# #ax.legend(loc='upper left')
# for s in ['top', 'right']:
#     ax.spines[s].set_visible(False)

# ax.yaxis.tick_right()
    
# ax.axhline(y = 0, color = 'black', linewidth = 1.3, alpha = .4)

# #ax.set_ylim(0, 50)
# #ax.legend(loc='upper left')
# for s in ['top', 'right','bottom','left']:
#     ax.spines[s].set_visible(False)

# ax.grid(False)
# ax.set_xticklabels(data_sub.index, fontfamily='serif', rotation=0)
# ax.margins(x=0) # remove white spaces next to margins

# ax.set_xlim(2008,2020)
# plt.xticks(np.arange(2008, 2021, 1))

# fig.text(0.13, 0.85, 'USA vs. India: When was content added?', fontsize=15, fontweight='bold', fontfamily='serif')
# fig.text(0.13, 0.58, 
# '''
# We know from our work above that Netflix is dominated by the USA & India.
# It would also be reasonable to assume that, since Netflix is an American
# compnany, Netflix increased content first in the USA, before 
# other nations. 

# That is exactly what we see here; a slow and then rapid
# increase in content for the USA, followed by Netflix 
# being launched to the Indian market in 2016.'''

# , fontsize=12, fontweight='light', fontfamily='serif')



# fig.text(0.13,0.15,"India", fontweight="bold", fontfamily='serif', fontsize=15, color='#b20710')
# fig.text(0.188,0.15,"|", fontweight="bold", fontfamily='serif', fontsize=15, color='black')
# fig.text(0.198,0.15,"USA", fontweight="bold", fontfamily='serif', fontsize=15, color='#221f1f')


# ax.tick_params(axis=u'both', which=u'both',length=0)


# plt.show()


# In[33]:


us_ind = df[(df['first_country'] == 'USA') | (df['first_country'] == 'India' )]

### STEFANOS: I didn't find a way to run the original with Modin. We're converting to Pandas and then
### back to Modin.
### ORIGINAL:
# data_sub = df.groupby('first_country')['year_added'].value_counts().unstack().fillna(0).loc[['USA','India']].cumsum(axis=0).T
### MODIFIED:

if "IREWR_WITH_MODIN" in os.environ and os.environ["IREWR_WITH_MODIN"] == "True":
  df = df._to_pandas()

data_sub = df.groupby('first_country')['year_added'].value_counts().unstack().fillna(0).loc[['USA','India']].cumsum(axis=0).T
if "IREWR_WITH_MODIN" in os.environ and os.environ["IREWR_WITH_MODIN"] == "True":
  df = pd.DataFrame(df)
  data_sub = pd.DataFrame(data_sub)


data_sub.insert(0, "base", np.zeros(len(data_sub)))

data_sub = data_sub.add(-us_ind['year_added'].value_counts()/2, axis=0)


# So the USA dominates. But is there a plot that can convey this in another way?

# In[34]:


# STEFANOS: Disable plotting

# fig, ax = plt.subplots(1, 1, figsize=(14, 6))
# color = ['#b20710','#221f1f'][::-1]
# hs_list = data_sub.columns
# hs_built = data_sub[hs]

# for i, hs in enumerate(hs_list):
#     if i == 0 : continue
#     ax.fill_between(hs_built.index, data_sub.iloc[:,i-1], data_sub.iloc[:,i], color=color[i-1])
    
# for s in ['top', 'right', 'bottom', 'left']:
#     ax.spines[s].set_visible(False)
# ax.set_axisbelow(True)
# ax.set_yticks([])
# #ax.legend(loc='upper left')
# ax.grid(False)

# fig.text(0.16, 0.76, 'USA vs. India: Stream graph of new content added', fontsize=15, fontweight='bold', fontfamily='serif')
# fig.text(0.16, 0.575, 
# '''
# Seeing the data displayed like this helps 
# us to realise just how much content is added in the USA.
# Remember, India has the second largest amount of
# content yet is dwarfed by the USA.'''

# , fontsize=12, fontweight='light', fontfamily='serif')

# fig.text(0.16,0.41,"India", fontweight="bold", fontfamily='serif', fontsize=15, color='#b20710')
# fig.text(0.208,0.41,"|", fontweight="bold", fontfamily='serif', fontsize=15, color='black')
# fig.text(0.218,0.41,"USA", fontweight="bold", fontfamily='serif', fontsize=15, color='#221f1f')


# ax.tick_params(axis=u'y', which=u'both',length=0)

# plt.show()


# # Lastly, we can view a wordcloud to get an overview of Netflix titles
# 
# 
# It is interesting to note that many films share the same key words in their titles.
# 
# 
# 
# Credit to Dmitry Uarov for figuring this visual out. His notebook is here:
# 
# https://www.kaggle.com/dmitryuarov/netflix-eda-with-plotly
# 
# 

# In[35]:


# STEFANOS: Uneeded
# from wordcloud import WordCloud
import random
from PIL import Image
import matplotlib

# Custom colour map based on Netflix palette
# STEFANOS: Disable plotting
# cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ['#221f1f', '#b20710'])

text = str(list(df['title'])).replace(',', '').replace('[', '').replace("'", '').replace(']', '').replace('.', '')

# STEFANOS: Can't find the icon
# mask = np.array(Image.open('input/netflix-icon-new/f6974e017d3f6196c4cbe284ee3eaf4e.png'))


# STEFANOS: Disable ML and plotting
# wordcloud = WordCloud(background_color = 'white', width = 500,  height = 200,colormap=cmap, max_words = 150, mask = mask).generate(text)

# plt.figure( figsize=(5,5))
# plt.imshow(wordcloud, interpolation = 'bilinear')
# plt.axis('off')
# plt.tight_layout(pad=0)
# plt.show()


# 
# # Thanks for reading!
# 
# # I hope you enjoyed my visuals 
# 
# # Please consider upvoting if you did
# 
# # Have a great day
# 
# 
# View more of my work:
# 
# **Exploratory Data Visualization - Student Performance**
# 
# https://www.kaggle.com/joshuaswords/data-visualisation-student-results
# 
# **Visuals & Modelling**
# 
# https://www.kaggle.com/joshuaswords/awesome-hr-data-visualization-prediction
# 
# **Visuals & Customer Segmentation**
# 
# https://www.kaggle.com/joshuaswords/data-visualization-clustering-mall-data
# 
# **March 2021 Tabular Playground Series**
# 
# https://www.kaggle.com/joshuaswords/tps-eda-model-march-2020
# 
# 
