#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis
# 
# In this notebook we will look into the speedtest data gathered from ookla and draw some useful insights from the dataset. Additional dataset would also be utilized to make some meaningful insights into the appropriately combined wholistic data

# ## File Exploration
# 
# In the following block we see all the files available for this kernel. The relevant libraries are also imported to process the data extracted from the dataset.

# In[3]:


# !pip install dias


# In[6]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import os
# STEFANOS: Conditionally import Modin Pandas
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
import regex as re # For String searches
# import plotly.graph_objects as go
# import plotly.express as px
# Input data files are available in the read-only "input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk(os.path.abspath('') +'/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[7]:


data = pd.read_csv('./input/speedtest-data-by-ookla/fixed_year_2021_quarter_03.csv')
data.head()


# In[8]:


# data.info()


# In[9]:


Mobile_df = pd.DataFrame([],columns=data.columns)
Broadband_df = pd.DataFrame([],columns=data.columns)

def col_name_corrections(df,correction_pair):
    if set(df.columns).intersection(set(correction_pair.keys())):
        df.rename(columns=correction_pair,inplace=True)
    return df

for dirname, _, filenames in os.walk('./input'):
    for filename in filenames:
        meta_info = filename.split('/')[-1]
        data = pd.read_csv(dirname+'/'+filename,thousands=r',').convert_dtypes()
        data = col_name_corrections(data,{'Number of Record':'Number of Records'})
        data['Year'] = np.int64(re.search('year_(.*)_quarter',meta_info).group(1))
        data['Quarter'] = np.int64(re.search('quarter_(.*).csv',meta_info).group(1))
        if 'mobile' in meta_info:
            Mobile_df = pd.concat([Mobile_df,data])
        else:
            Broadband_df = pd.concat([Broadband_df,data]) 
print(Broadband_df.shape)
print(Mobile_df.shape)
Mobile_df = Mobile_df.astype({'Year':np.int64,'Quarter':np.int64})
Broadband_df = Broadband_df.astype({'Year':np.int64,'Quarter':np.int64})
Mobile_df.sort_values(by=['Year','Quarter'],ascending=[True,True],inplace=True)
Broadband_df.sort_values(by=['Year','Quarter'],ascending=[True,True],inplace=True)


# It can be seen that there are more broadband rows than Mobile rows. This is a point that should be noted each row corresponds to a country's statistics in the particular year and quarter. Missing data indicates lack of speed test data from the country.

# In[10]:


# Mobile_df.info()


# # -- STEFANOS -- Replicate Data

# In[11]:


factor = 2000
Mobile_df = pd.concat([Mobile_df]*factor, ignore_index=True)
# Mobile_df.info()


# In[12]:


factor = 1000
Broadband_df = pd.concat([Broadband_df]*factor, ignore_index=True)
# Broadband_df.info()


# In[13]:


Mobile_df.head()


# In[14]:


unique_countries_broadband = Broadband_df.groupby('Name').count()
unique_countries_broadband.head()


# In[15]:


unique_countries_mobile = Mobile_df.groupby('Name').count()
unique_countries_mobile.head()


# ## Insights
# 
# The following countries do have mobile speedtest data for all the years and quarters, thereby making less than 10 reports (one per quarter).

# In[16]:


# Check for missing values
Mobile_df.isna().any()


# In[17]:


Broadband_df.isna().any()


# In[18]:


unique_countries_mobile[unique_countries_mobile.Year < 10]['Year']


# In[19]:


unique_countries_broadband[unique_countries_broadband.Year < 10]['Year']


# ## Raw Download Speed Visualization
# 
# This visualization can be used to show change of values per country. The improvement values cannot be understood by laymen because an improvement of 50 Kbps **national average** (given) means differnt things to different countries based on economy, population, GDP, Infrastructure, etc.

# In[20]:





# In[41]:


# DIAS_VERBOSE
Mobile_Stats = Mobile_df.groupby('Name').agg(
    Change_Download=('Avg. Avg D Kbps', lambda x: list(x)[-1] - list(x)[0]),
    Change_Upload=('Avg. Avg U Kbps', lambda x: list(x)[-1] - list(x)[0]),
    Change_Latency=('Avg Lat Ms', lambda x: list(x)[-1] - list(x)[0])
)
Broadband_Stats = Broadband_df.groupby('Name').agg(
    Change_Download=('Avg. Avg D Kbps', lambda x: list(x)[-1] - list(x)[0]),
    Change_Upload=('Avg. Avg U Kbps', lambda x: list(x)[-1] - list(x)[0]),
    Change_Latency=('Avg Lat Ms', lambda x: list(x)[-1] - list(x)[0])
)
# fig = px.histogram(Mobile_Stats['Change_Download'],title='Frequency distribution of Mobile Speed change',
#                    labels={'count':'Frequency','value':'$\Delta Speed (Kpbs)$','variable':'property'},
#                    nbins=100)
# fig.show()

# fig = px.histogram(Broadband_Stats['Change_Download'],title='Frequency distribution of Broadband Speed change',
#                    labels={'count':'Frequency','value':'$\Delta Speed (Kpbs)$','variable':'property'},
#                    nbins=100)
# fig.show()
Total_Stats = pd.concat([Broadband_Stats['Change_Download'],Mobile_Stats['Change_Download']],axis=1)
Total_Stats.columns=['Mobile','Broadband']

# STEFANOS: Disable plotting
# fig = go.Figure(data=[go.Histogram(x=Broadband_Stats['Change_Download'],opacity=0.65,name='Broadband')])
# fig.add_trace(go.Histogram(x=Mobile_Stats['Change_Download'],opacity=0.65,name='Mobile'))
# fig.update_layout(barmode='overlay',
#                   title='Frequency Distribution of Speed change',
#                   xaxis_title="$\Delta\ Speed\ (Kbps)$", yaxis_title="Number of Countries",
#                   legend_title='Color')
# fig.show()


# It can see that most of the countries changed between -5000 Kbps to 5000 kbps. A common graph for all the countries is possible but makes it difficult to understand. Therefore it is better we split the countries into different visualizations, for seperate degrees of change

# In[23]:


# STEFANOS: Disable plotting
# px.bar(Mobile_Stats,y='Change_Download',labels={'Name':'Country','Change_Download':'Observed Change'},title='Summary of all changes 2020 Q1 - 2022 Q2 ')


# In[24]:


#ImprovedCountries_M = Mobile_Stats[(Mobile_Stats['Change_Download'] < 3000) &
#                                (Mobile_Stats['Change_Download'] >0)]
#px.bar(ImprovedCountries_M,y='Change_Download',labels={'Name':'Country','Change_Download':'Improved Download Speed'},title='Countries that improved download speeds')

ImprovedCountries_B = Broadband_Stats[(Broadband_Stats['Change_Download'] < 3000) &
                                (Broadband_Stats['Change_Download'] > 0)]

# STEFANOS: Disable plotting
# fig = go.Figure()
# fig.add_trace(go.Bar(x=ImprovedCountries_B.index,y=ImprovedCountries_B['Change_Download'],opacity=0.6,name='Broadband'))
# fig.add_trace(go.Bar(x=ImprovedCountries_B.index,y=Mobile_Stats.query("index in @ImprovedCountries_B.index")['Change_Download'],opacity=0.6,name='Mobile'))
# fig.update_layout(barmode='group',
#                   title='Bar Chart of Speed change',
#                   xaxis_title="Country", yaxis_title="Improved Speed (Kbps)",
#                   legend_title='Color')
# fig.show()


# In[25]:


#ImprovedCountries2 = Mobile_Stats[(Mobile_Stats['Change_Download'] >= 10000)]
#px.bar(ImprovedCountries2,y='Change_Download',labels={'Name':'Country','Change_Download':'Improved Download Speed'},title='Countries that improved download speeds')
#ImprovedCountries_M = Mobile_Stats[(Mobile_Stats['Change_Download'] < 8000) &
#                                (Mobile_Stats['Change_Download'] >3000)]
#px.bar(ImprovedCountries_M,y='Change_Download',labels={'Name':'Country','Change_Download':'Improved Download Speed'},title='Countries that improved download speeds')

ImprovedCountries_B = Broadband_Stats[(Broadband_Stats['Change_Download'] < 8000) &
                                (Broadband_Stats['Change_Download'] > 3000)]
# STEFANOS: Disable plotting
# fig = go.Figure()
# fig.add_trace(go.Bar(x=ImprovedCountries_B.index,y=ImprovedCountries_B['Change_Download'],opacity=0.6,name='Broadband'))
# fig.add_trace(go.Bar(x=ImprovedCountries_B.index,y=Mobile_Stats.query("index in @ImprovedCountries_B.index")['Change_Download'],opacity=0.6,name='Mobile'))
# fig.update_layout(barmode='group',
#                   title='Bar Chart of Speed change',
#                   xaxis_title="Country", yaxis_title="Improved Speed (Kbps)",
#                   legend_title='Color')
# fig.show()


# In[26]:


ImprovedCountries_B = Broadband_Stats[(Broadband_Stats['Change_Download'] < 16000) &
                                (Broadband_Stats['Change_Download'] > 8000)]
# STEFANOS: Disable plotting
# fig = go.Figure()
# fig.add_trace(go.Bar(x=ImprovedCountries_B.index,y=ImprovedCountries_B['Change_Download'],opacity=0.6,name='Broadband'))
# fig.add_trace(go.Bar(x=ImprovedCountries_B.index,y=Mobile_Stats.query("index in @ImprovedCountries_B.index")['Change_Download'],opacity=0.6,name='Mobile'))
# fig.update_layout(barmode='group',
#                   title='Bar Chart of Speed change',
#                   xaxis_title="Country", yaxis_title="Improved Speed (Kbps)",
#                   legend_title='Color')
# fig.show()


# In[27]:


ImprovedCountries_B = Broadband_Stats[(Broadband_Stats['Change_Download'] < 60000) &
                                (Broadband_Stats['Change_Download'] > 16000)]
# STEFANOS: Disable plotting
# fig = go.Figure()
# fig.add_trace(go.Bar(x=ImprovedCountries_B.index,y=ImprovedCountries_B['Change_Download'],opacity=0.6,name='Broadband'))
# fig.add_trace(go.Bar(x=ImprovedCountries_B.index,y=Mobile_Stats.query("index in @ImprovedCountries_B.index")['Change_Download'],opacity=0.6,name='Mobile'))
# fig.update_layout(barmode='group',
#                   title='Bar Chart of Speed change',
#                   xaxis_title="Country", yaxis_title="Improved Speed (Kbps)",
#                   legend_title='Color')
# fig.show()


# In[28]:


ImprovedCountries_B = Broadband_Stats[(Broadband_Stats['Change_Download'] >= 60000)]
# STEFANOS: Disable plotting
# fig = go.Figure()
# fig.add_trace(go.Bar(x=ImprovedCountries_B.index,y=ImprovedCountries_B['Change_Download'],opacity=0.6,name='Broadband'))
# fig.add_trace(go.Bar(x=ImprovedCountries_B.index,y=Mobile_Stats.query("index in @ImprovedCountries_B.index")['Change_Download'],opacity=0.6,name='Mobile'))
# fig.update_layout(barmode='group',
#                   title='Bar Chart of Speed change',
#                   xaxis_title="Country", yaxis_title="Improved Speed (Kbps)",
#                   legend_title='Color')
# fig.show()


# In[29]:


#DeterioratedSpeeds = Mobile_Stats[(Mobile_Stats['Change_Download'] < 0 )]
#px.bar(DeterioratedSpeeds,y='Change_Download',labels={'Name':'Country','Change_Download':'Improved Download Speed'},title='Decreasing Countries\' download speeds')
ImprovedCountries_B = Broadband_Stats[(Broadband_Stats['Change_Download'] < 0)]
Countries = ImprovedCountries_B.index

# STEFANOS: Disable plotting
# fig = go.Figure()
# fig.add_trace(go.Bar(x=ImprovedCountries_B.index,y=ImprovedCountries_B['Change_Download'],opacity=0.6,name='Broadband'))
# fig.add_trace(go.Bar(x=ImprovedCountries_B.index,y=Mobile_Stats.query("index in @ImprovedCountries_B.index")['Change_Download'],opacity=0.6,name='Mobile'))
# fig.update_layout(barmode='group',
#                   title='Bar Chart of Speed change',
#                   xaxis_title="Country", yaxis_title="Improved Speed (Kbps)",
#                   legend_title='Color')
# fig.show()


# In[30]:


Mobile_Stats.sort_values(by=['Change_Download'])


# In[31]:


Broadband_Stats.sort_values(by=['Change_Download'])


# Different graphs are used to show different degrees of Average download speed change for each country. These are the capacities by which countries have changed. 
# 
# From the above graphs it can be seen that **China** has improved the most average (boardband) download speed and **Antarctica** has lost the most average download internet speed (broadband). Meanwhile **Korea** has the highest improvement in Mobile internet download speed, while **Cook Islands** has lost the most average mobile download speed
# 
# These metrics can be misleading as they show average speed change for the whole country. The infrastructure investment/deterioration of the country can only be known after the number is multiplied by population of the country.

# ## Percentage Download speed Visualization
# 
# In this view, we have normalized all improvements or depreciation to the original value, therefore large countries such as china which already have large infrastructure will be given less weightage and small countries that are developing would be given more preference. This view can be used everywhere as the values are normalized to percentages. Therefore larger internet speed nations will be given lower priority to the smaller speed nations. This is NOT a metric of a nation's total bandwidth as that will require denormalization with the nation's population. This metric can be used to compare the rate of national speed improvement between nations.

# In[32]:


Mobile_Stats_relative = Mobile_df.groupby('Name').agg(
    Change_Download=('Avg. Avg D Kbps', lambda x: (list(x)[-1] - list(x)[0])/list(x)[0]),
    Change_Upload=('Avg. Avg U Kbps', lambda x: (list(x)[-1] - list(x)[0])/list(x)[0]),
    Change_Latency=('Avg Lat Ms', lambda x: (list(x)[-1] - list(x)[0])/list(x)[0])
)
Broadband_Stats_relative = Broadband_df.groupby('Name').agg(
    Change_Download=('Avg. Avg D Kbps', lambda x: (list(x)[-1] - list(x)[0])/list(x)[0]),
    Change_Upload=('Avg. Avg U Kbps', lambda x: (list(x)[-1] - list(x)[0])/list(x)[0]),
    Change_Latency=('Avg Lat Ms', lambda x: (list(x)[-1] - list(x)[0])/list(x)[0])
)


# In[33]:


ImprovedCountries_B = Broadband_Stats_relative[(Broadband_Stats_relative['Change_Download'] >= 2)]
# STEFANOS: Disable plotting
# fig = go.Figure()
# fig.add_trace(go.Bar(x=ImprovedCountries_B.index,y=ImprovedCountries_B['Change_Download'],opacity=0.6,name='Broadband'))
# fig.add_trace(go.Bar(x=ImprovedCountries_B.index,y=Mobile_Stats_relative.query("index in @ImprovedCountries_B.index")['Change_Download'],opacity=0.6,name='Mobile'))
# fig.update_layout(barmode='group',
#                   title='Bar Chart of Relative Speed change',
#                   xaxis_title="Country", yaxis_title="Relative Improvement of Speed",
#                   legend_title='Color')
# fig.show()


# In[34]:


ImprovedCountries_B = Broadband_Stats_relative[(Broadband_Stats_relative['Change_Download'] >= 1) & (Broadband_Stats_relative['Change_Download'] < 2)]
# STEFANOS: Disable plotting
# fig = go.Figure()
# fig.add_trace(go.Bar(x=ImprovedCountries_B.index,y=ImprovedCountries_B['Change_Download'],opacity=0.6,name='Broadband'))
# fig.add_trace(go.Bar(x=ImprovedCountries_B.index,y=Mobile_Stats_relative.query("index in @ImprovedCountries_B.index")['Change_Download'],opacity=0.6,name='Mobile'))
# fig.update_layout(barmode='group',
#                   title='Bar Chart of Relative Speed change',
#                   xaxis_title="Country", yaxis_title="Relative Improvement of Speed",
#                   legend_title='Color')
# fig.show()


# In[35]:


ImprovedCountries_B = Broadband_Stats_relative[(Broadband_Stats_relative['Change_Download'] >= 0.5) & (Broadband_Stats_relative['Change_Download'] < 1)]
# STEFANOS: Disable plotting
# fig = go.Figure()
# fig.add_trace(go.Bar(x=ImprovedCountries_B.index,y=ImprovedCountries_B['Change_Download'],opacity=0.6,name='Broadband'))
# fig.add_trace(go.Bar(x=ImprovedCountries_B.index,y=Mobile_Stats_relative.query("index in @ImprovedCountries_B.index")['Change_Download'],opacity=0.6,name='Mobile'))
# fig.update_layout(barmode='group',
#                   title='Bar Chart of Relative Speed change',
#                   xaxis_title="Country", yaxis_title="Relative Improvement of Speed",
#                   legend_title='Color')
# fig.show()


# In[36]:


ImprovedCountries_B = Broadband_Stats_relative[(Broadband_Stats_relative['Change_Download'] >= 0.2) & (Broadband_Stats_relative['Change_Download'] < 0.5)]
# STEFANOS: Disable plotting
# fig = go.Figure()
# fig.add_trace(go.Bar(x=ImprovedCountries_B.index,y=ImprovedCountries_B['Change_Download'],opacity=0.6,name='Broadband'))
# fig.add_trace(go.Bar(x=ImprovedCountries_B.index,y=Mobile_Stats_relative.query("index in @ImprovedCountries_B.index")['Change_Download'],opacity=0.6,name='Mobile'))
# fig.update_layout(barmode='group',
#                   title='Bar Chart of Relative Speed change',
#                   xaxis_title="Country", yaxis_title="Relative Improvement of Speed",
#                   legend_title='Color')
# fig.show()


# In[37]:


ImprovedCountries_B = Broadband_Stats_relative[(Broadband_Stats_relative['Change_Download'] >= 0) & (Broadband_Stats_relative['Change_Download'] < 0.2)]
# STEFANOS: Disable plotting
# fig = go.Figure()
# fig.add_trace(go.Bar(x=ImprovedCountries_B.index,y=ImprovedCountries_B['Change_Download'],opacity=0.6,name='Broadband'))
# fig.add_trace(go.Bar(x=ImprovedCountries_B.index,y=Mobile_Stats_relative.query("index in @ImprovedCountries_B.index")['Change_Download'],opacity=0.6,name='Mobile'))
# fig.update_layout(barmode='group',
#                   title='Bar Chart of Relative Speed change',
#                   xaxis_title="Country", yaxis_title="Relative Improvement of Speed",
#                   legend_title='Color')
# fig.show()


# In[38]:


ImprovedCountries_B = Broadband_Stats_relative[(Broadband_Stats_relative['Change_Download'] < 0)]
# STEFANOS: Disable plotting
# fig = go.Figure()
# fig.add_trace(go.Bar(x=ImprovedCountries_B.index,y=ImprovedCountries_B['Change_Download'],opacity=0.6,name='Broadband'))
# fig.add_trace(go.Bar(x=ImprovedCountries_B.index,y=Mobile_Stats_relative.query("index in @ImprovedCountries_B.index")['Change_Download'],opacity=0.6,name='Mobile'))
# fig.update_layout(barmode='group',
#                   title='Bar Chart of Relative Speed change',
#                   xaxis_title="Country", yaxis_title="Relative Improvement of Speed",
#                   legend_title='Color')
# fig.show()


# In[39]:


Broadband_Stats_relative.sort_values(by=['Change_Download'])


# In[40]:


Mobile_Stats_relative.sort_values(by=['Change_Download'])


# In this view we have **Saint Pierre and Miquelon** improving the most relative to its baseline speed and **Wallis and Futuna Islands** losing almost all of its original national average download speed for mobile devices. On the other hand, we have **Antarctica** losing 0.82x its original average speed while Tongo improve its speed by 5x for broadband downloads. Either ends of this view of data would be filled with countries that have low download speeds to begin with, as in that case a minute improvement would also be amplified due to its relative nature.
