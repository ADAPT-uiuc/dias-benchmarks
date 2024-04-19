#!/usr/bin/env python
# coding: utf-8

# # <div style="text-align: center; background-color: #39c5ff;font-size:200%; font-family:Rubik; color: #ffffff; padding: 20px; line-height: 1;border-radius:10px; border: 5px solid #3f484b;"><b>Kaggle <br> Survey 2022</b></div>

# <a id="top"></a>
# <div class="list-group" id="list-tab" role="tablist">
#     <h3 style="text-align: left; background-color: #3f484b; font-family:Rubik; color: #F5F5F1; padding: 14px; line-height: 1; border-radius:10px">Table of Contents</h3>
#     
#    * [1. Imports](#1)
#    * [2. Data Cleaning](#2)
#    * [3. Exploratory Data Analysis](#3)
#     - [3.1. Univariate Analysis](#3.1)
#         - [3.1.1. Gender Countplot](#3.1.1)
#         - [3.1.2. Duration Histogram](#3.1.2)
#         - [3.1.3. Age Countplot](#3.1.3)
#         - [3.1.4. Top 10 Countries Count](#3.1.4)
#         - [3.1.5. World Map](#3.1.5)
#         - [3.1.6. Student Donut Chart](#3.1.6)
#         - [3.1.7. Educational Background Donut Chart](#3.1.7)
#         - [3.1.8. ML Experience Donut Chart](#3.1.8)
#         - [3.1.9. Yearly Compensation Donut Chart](#3.1.9)
#     - [3.2. Multivariate Analysis](#3.2)
#         - [3.2.1. Duration X Gender Histplot](#3.2.1)
#         - [3.2.2. Country X Gender Treemap](#3.2.2)
#         - [3.2.3. Duration with Age](#3.2.3)
#         - [3.2.4. Education X Gender Treemap](#3.2.4)
#         - [3.2.5. Country X Age Density Plot](#3.2.5)
#         - [3.2.6. Gender X Student Sunburst](#3.2.6)
#    * [4. Work In Progress](#4)

# <a id="1"></a>
# # <div style="font-size:40px; text-align: center; background-color: #39c5ff; font-family:Rubik; color: #ffffff; padding: 14px; line-height: 1;border-radius:20px; border: 4px solid #3f484b;"><b> 1. Imports </b></div>

# In[1]:


import os
# STEFANOS: Conditionally import Modin Pandas
import pandas as pd
import numpy as np



colors = ['gold', 'mediumturquoise', 'darkorange', 'lightgreen']
kaggle_colors = ['#39c5ff', '#ffffff', '#f2f2f2', '#9ca3a4', '#3f484b']
font = 'Rubik'


# In[2]:


df = pd.read_csv('input/kaggle-survey-2022/kaggle_survey_2022_responses.csv')


# In[3]:


df


# <a id="2"></a>
# # <div style="font-size:40px; text-align: center; background-color: #39c5ff; font-family:Rubik; color: #ffffff; padding: 14px; line-height: 1;border-radius:20px; border: 4px solid #3f484b;"><b> 2. Data Cleaning </b></div>

# In[4]:


df = df[1:]


# # -- STEFANOS -- Replicate Data

# In[5]:


factor = 10
df = pd.concat([df]*factor, ignore_index=True)
# df.info()


# In[6]:


df['Duration (in seconds)'] = df['Duration (in seconds)'].astype('int')


# In[7]:


df.loc[df["Q3"] == "Nonbinary", "Q3"] = "Others"
df.loc[df["Q3"] == "Prefer not to say", "Q3"] = "Others"
df.loc[df["Q3"] == "Prefer to self-describe", "Q3"] = "Others"


# <a id="3"></a>
# # <div style="font-size:40px; text-align: center; background-color: #39c5ff; font-family:Rubik; color: #ffffff; padding: 14px; line-height: 1;border-radius:20px; border: 4px solid #3f484b;"><b> 3. Exploratory Data Analysis </b></div>

# <a id="3.1"></a>
# <h3 style="font-size:25px; text-align: left;background-color: #39c5ff; font-family:Rubik; color: #ffffff; padding: 16px; line-height: 1; border-radius:10px; border: 3px solid #3f484b;"> 3.1 Univariate Analysis</h3>

# <a id="3.1.1"></a>
# <span style="font-size:20px; color:#39c5ff; font-family:Rubik;"><i> 3.1.1 Gender Countplot </i></span>

# In[8]:


gender_count = df.groupby(['Q3']).size().reset_index().rename(columns={0: 'count'})

# STEFANOS: Disable plotting
# fig=px.bar(data_frame=gender_count, x='Q3', y='count',color='Q3',template='plotly_dark',color_discrete_sequence=['#39c5ff','#F5F5F1'],
#                  text_auto=True,barmode='stack',
#           title="Gender Countplot")

# fig.update_traces(textfont_size=20, textangle=0, textposition="outside", cliponaxis=False)
# fig.update_traces( marker_line_color='#3f484b',
#                   marker_line_width=3.5)

# fig.update_layout(
#     title="Gender Countplot",
#     xaxis_title="Genders",
#     yaxis_title="Count",
#     legend_title="Genders",
#     font=dict(
#         family="Rubik",
#         size=18
#     )
# )

# fig.show()


# <a id="3.1.2"></a>
# <span style="font-size:20px; color:#39c5ff; font-family:Rubik;"> <i>3.1.2 Duration Histogram</i></span>

# In[9]:


duration_count = df.groupby(['Duration (in seconds)']).size().reset_index().rename(columns={0: 'count'})
duration_count = duration_count[:1000]

# STEFANOS: Disable plotting
# fig=px.histogram(data_frame=duration_count, x='Duration (in seconds)', y='count',template='plotly_dark',color_discrete_sequence=['#39c5ff','#F5F5F1'],
#                  text_auto=True,barmode='stack',
#           title="Duration Histogram")

# fig.update_traces(textfont_size=20, textangle=0, textposition="outside", cliponaxis=False)
# fig.update_traces( marker_line_color='#3f484b',
#                   marker_line_width=3.5)

# fig.update_layout(
#     title="Duration Histogram",
#     font=dict(
#         family="Rubik",
#         size=14
#     )
# )

# fig.show()


# <a id="3.1.3"></a>
# <span style="font-size:20px; color:#39c5ff; font-family:Rubik;"> <i>3.1.3 Age Countplot</i></span>

# In[10]:


age_count = df.groupby(['Q2']).size().reset_index().rename(columns={0: 'count'})

# STEFANOS: Disable plotting
# fig=px.bar(data_frame=age_count, x='Q2', y='count',color='Q2',template='plotly_dark',color_discrete_sequence=['#39c5ff','#F5F5F1'],
#                  text_auto=True,barmode='stack',
#           title="Age Countplot")

# fig.update_traces(textfont_size=20, textangle=0, textposition="outside", cliponaxis=False)
# fig.update_traces( marker_line_color='#3f484b',
#                   marker_line_width=3.5)

# fig.update_layout(
#     title="Age Countplot",
#     xaxis_title="Age Groups",
#     yaxis_title="Count",
#     legend_title="Age Groups",
#     font=dict(
#         family="Rubik",
#         size=12
#     )
# )

# fig.show()


# <a id="3.1.4"></a>
# <span style="font-size:20px; color:#39c5ff; font-family:Rubik;"> <i>3.1.4 Top 10 Countries Count</i></span>

# In[11]:


country_count = df.groupby(['Q4']).size().reset_index().rename(columns={0: 'count'})
country_count = country_count.sort_values(by=['count'], ascending=False).reset_index(drop=True)
country_count = country_count[:10]

# STEFANOS: Disable plotting
# fig=px.bar(data_frame=country_count,x='Q4', y='count',color='Q4',template='plotly_dark',color_discrete_sequence=['#39c5ff','#F5F5F1'],
#                  text_auto=True,barmode='stack',
#           title="Country Countplot")

# fig.update_traces(textfont_size=20, textangle=0, textposition="outside", cliponaxis=False)
# fig.update_traces( marker_line_color='#3f484b',
#                   marker_line_width=3.5)

# fig.update_layout(
#     title="Top 10 Countries Plot",
#     xaxis_title="Countries",
#     yaxis_title="Count",
#     legend_title="Countries",
#     font=dict(
#         family="Rubik",
#         size=13
#     )
# )

# fig.show()


# <a id="3.1.5"></a>
# <span style="font-size:20px; color:#39c5ff; font-family:Rubik;"> <i>3.1.5 World Map</i></span>

# In[12]:


country_df = df.groupby(['Q4']).size().reset_index().rename(columns={0: 'count'})

# STEFANOS: Disable plotting
# fig = px.choropleth(country_df, locations="Q4", color="count", 
#                     locationmode='country names',
#                     range_color=[0,5000],
#                     color_continuous_scale=[(0, "#ffffff"), (1, '#39c5ff')],
#                     template='plotly_dark'
#                    )

# fig.update_layout(
#     title="World Map",
#     font=dict(
#         family="Rubik",
#         size=14
#     )
# )

# fig.show()


# <a id="3.1.6"></a>
# <span style="font-size:20px; color:#39c5ff; font-family:Rubik;"> <i>3.1.6 Student Donut Chart</i></span>

# In[13]:


student_count = df.groupby(['Q5']).size().reset_index().rename(columns={0: 'count'})

# STEFANOS: Disable plotting
# fig = go.Figure(
#     data=[go.Pie(labels=student_count['Q5'], values=student_count['count'], hole=.4)])

# fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,
#                   marker=dict(colors=colors, line=dict(color='#000000', width=2)))

# fig.update_layout(
#     title="Student Count",
#     template='plotly_dark',
#     font=dict(
#         family="Rubik",
#         size=18
#     )
# )

# fig.show()


# <a id="3.1.7"></a>
# <span style="font-size:20px; color:#39c5ff; font-family:Rubik;"> <i>3.1.7 Educational Background Donut Chart</i></span>

# In[14]:


degree_count = df.groupby(['Q8']).size().reset_index().rename(columns={0: 'count'})
degree_count.loc[degree_count["Q8"] == "Some college/university study without earning a bachelor’s degree", "Q8"] = "College Without Bachelors"

# STEFANOS: Disable plotting
# fig = go.Figure(
#     data=[go.Pie(labels=degree_count['Q8'], values=degree_count['count'], hole=.4)])

# fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,
#                   marker=dict(colors=colors, line=dict(color='#000000', width=2)))

# # fig.update_layout(template='plotly_dark')

# fig.update_layout(
#     title="Educational Background",
#     template='plotly_dark',
#     font=dict(
#         family="Rubik",
#         size=14
#     )
# )

# fig.show()


# <a id="3.1.8"></a>
# <span style="font-size:20px; color:#39c5ff; font-family:Rubik;"> <i>3.1.8 ML Experience Donut Chart</i></span>

# In[15]:


experience_count = df.groupby(['Q16']).size().reset_index().rename(columns={0: 'count'})

# STEFANOS: Disable plotting
# fig = go.Figure(
#     data=[go.Pie(labels=experience_count['Q16'], values=experience_count['count'], hole=.4)])

# fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,
#                   marker=dict(colors=colors, line=dict(color='#000000', width=2)))

# # fig.update_layout(template='plotly_dark')

# fig.update_layout(
#     title="ML Experience",
#     template='plotly_dark',
#     font=dict(
#         family="Rubik",
#         size=14
#     )
# )

# fig.show()


# <a id="3.1.9"></a>
# <span style="font-size:20px; color:#39c5ff; font-family:Rubik;"> <i>3.1.9 Yearly Compensation Donut Chart</i></span>

# In[16]:


annual_income_df = df.groupby(['Q29']).size().reset_index().rename(columns={0: 'count'})

# STEFANOS: Disable plotting
# fig = go.Figure(
#     data=[go.Pie(labels=annual_income_df['Q29'], values=annual_income_df['count'], hole=.4)])

# fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,
#                   marker=dict(colors=colors, line=dict(color='#000000', width=2)))

# # fig.update_layout(template='plotly_dark')

# fig.update_layout(
#     title="Yearly Compensation in USD",
#     template='plotly_dark',
#     font=dict(
#         family="Rubik",
#         size=14
#     )
# )

# fig.show()


# <a id="3.2"></a>
# <h3 style="font-size:25px; text-align: left;background-color: #39c5ff; font-family:Rubik; color: #ffffff; padding: 16px; line-height: 1; border-radius:10px; border: 3px solid #3f484b;"> 3.2 Multivariate Analysis</h3>

# <a id="3.2.1"></a>
# <span style="font-size:20px; color:#39c5ff; font-family:Rubik;"> <i>3.2.1 Duration X Gender Histplot</i></span>

# In[17]:


gender_duration_count = df.groupby(['Duration (in seconds)', 'Q3']).size().reset_index().rename(columns={0: 'count'})
gender_duration_count = gender_duration_count[:3000]

# STEFANOS: Disable plotting
# hist_colors = ['#39c5ff', '#ffffff', 'orange']

# fig = px.histogram(gender_duration_count, x="Duration (in seconds)", y="count", color='Q3',
#                    marginal="box", color_discrete_sequence=hist_colors,
#                    hover_data=gender_duration_count.columns,
#                   template='plotly_dark')

# fig.update_layout(
#     title="Duration X Gender",
#     xaxis_title="Duration (in seconds)",
#     yaxis_title="Count",
#     legend_title="Gender",
#     font=dict(
#         family="Rubik",
#         size=16
#     )
# )

# fig.show()


# <a id="3.2.2"></a>
# <span style="font-size:20px; color:#39c5ff; font-family:Rubik;"> <i>3.2.2 Country X Gender TreeMap</i></span>

# In[18]:


country_gender = df.groupby(['Q4', 'Q3']).size().reset_index().rename(columns={0: 'count'})

# STEFANOS: Disable plotting
# fig = px.treemap(country_gender, values='count', path=['Q4', 'Q3'], template='plotly_dark',
#                 title="Country X Gender TreeMap")
# fig.update_traces(textinfo="label+percent parent")

# fig.show()


# <a id="3.2.3"></a>
# <span style="font-size:20px; color:#39c5ff; font-family:Rubik;"> <i>3.2.3 Duration with Age</i></span>

# In[19]:


age_duration_df = df[df['Duration (in seconds)']<2000].groupby('Q2')['Duration (in seconds)'].mean().round(1).reset_index()

# STEFANOS: Disable plotting
# fig=px.bar(data_frame=age_duration_df, x='Q2', y='Duration (in seconds)',
#            color='Q2',template='plotly_dark',color_discrete_sequence=['#39c5ff','#F5F5F1'],
#                  text_auto=True,barmode='stack',
#           title="Age Countplot")

# fig.update_traces(textfont_size=20, textangle=0, textposition="outside", cliponaxis=False)
# fig.update_traces( marker_line_color='#3f484b',
#                   marker_line_width=3.5)

# fig.update_layout(
#     title="Duration with Age",
#     xaxis_title="Age Groups",
#     yaxis_title="Duration (in seconds)",
#     legend_title="Age Groups",
#     font=dict(
#         family="Rubik",
#         size=12
#     )
# )

# fig.show()


# <a id="3.2.4"></a>
# <span style="font-size:20px; color:#39c5ff; font-family:Rubik;"> <i>3.2.4 Education X Gender TreeMap</i></span>

# In[20]:


education_gender = df.groupby(['Q8', 'Q3']).size().reset_index().rename(columns={0: 'count'})

# STEFANOS: Disable plotting
# fig = px.treemap(education_gender, values='count', path=['Q8', 'Q3'], template='plotly_dark',
#                 title="Education X Gender TreeMap")
# fig.update_traces(textinfo="label+percent parent")

# fig.show()


# <a id="3.2.5"></a>
# <span style="font-size:20px; color:#39c5ff; font-family:Rubik;"> <i>3.2.5 Countries X Age Density Plot</i></span>

# In[21]:


## Not Including India bcz it acts as an outlier here
top_9_countries = ['United States of America','Other', 'Brazil',
                   'Nigeria', 'Pakistan', 'Japan', 'China', 'Egypt', 'Mexico']

top_9_countries_df = df[df['Q4'].isin(top_9_countries)]
# STEFANOS: Disable plotting
# fig = px.density_heatmap(
#     top_9_countries_df, x='Q4', y='Q2',
#     marginal_x='histogram', marginal_y='histogram', histfunc='count'
# )

# fig.update_yaxes(categoryorder='array', categoryarray= ['18-21','22-24','25-29','30-34', '35-39',
#                                                         '40-44', '45-49', '50-54', '55-59', '60-69', '70+'])

# fig.update_layout(
#     title="Countires X Age Groups Density Plot",
#     xaxis_title="Countries",
#     yaxis_title="Age Groups",
#     font=dict(
#         family="Rubik",
#         size=14
#     )
# )

# fig.show()


# <a id="3.2.6"></a>
# <span style="font-size:20px; color:#39c5ff; font-family:Rubik;"> <i>3.2.6 Gender X Student Sunburst</i></span>

# In[22]:


gender_student = df.groupby(['Q3', 'Q5']).size().reset_index().rename(columns={0: 'count'})

# STEFANOS: Disable plotting
# fig = px.sunburst(gender_student, values='count', path=['Q3', 'Q5'], template='plotly_dark',
#                 title="Gender X Student Sunburst")
# fig.update_traces(textinfo="label+percent parent")

# fig.show()


# <a id="4"></a>
# # <div style="font-size:40px; text-align: center; background-color: #39c5ff; font-family:Rubik; color: #ffffff; padding: 14px; line-height: 1;border-radius:20px; border: 4px solid #3f484b;"><b> Work In Progress </b></div>
