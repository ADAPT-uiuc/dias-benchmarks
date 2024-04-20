#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis
# 
# 
# Exploratory Data Analysis on dataset 'SampleSuperstore' to find out the patterns in data and to analyse the business trends in order to determine the weak areas that needs to be worked on in order to make more profit

# # Importing Libraries

# In[1]:


import matplotlib.pyplot as plt
import seaborn as sns
import os
# STEFANOS: Conditionally import Modin Pandas
import pandas as pd


# # Reading the data

# In[2]:


data=pd.read_csv(os.path.abspath('') + "/input/superstore/SampleSuperstore.csv")


# # -- STEFANOS -- Replicate Data

# In[3]:


factor = 800
data = pd.concat([data]*factor)
# data.info()


# In[4]:


data.head()


# In[5]:


# data.info()


# In[6]:


data.duplicated().sum()


# #### So, there are 17 duplicate entries and let us drop them.

# In[7]:


# STEFANOS: Disable this. It undos the data replication.
# data.drop_duplicates(inplace= True)


# In[8]:


# data.info()


# # Dropping unwanted columns 

# In[9]:


data.drop(["Postal Code"], axis=1,inplace= True)


# In[10]:


# data.info()


# # Correlation between the data

# In[11]:


data.corr()


# In[12]:


# STEFANOS: Disable plotting
# sns.heatmap(data.corr(), annot=True)


# # EDA

# ### Analysing the different kinds of Shipping Modes, Segments and categories mentioned in the data

# ## Shipping Mode

# In[13]:


data["Ship Mode"].value_counts()


# In[14]:


# STEFANOS: Disable plotting
# sns.countplot(x= data['Ship Mode'])


# ## Different Segments

# In[15]:


data["Segment"].value_counts()


# In[16]:


# STEFANOS: Disable plotting
# sns.countplot(x= data['Segment'])


# ## Categories of the items

# In[17]:


data["Category"].value_counts()


# In[18]:


# STEFANOS: Disable plotting
# sns.countplot(x= data['Category'])


# #### From the above plot we can conclude that Office Supplies Category has highest number of sales. Now let us see the sub-categories as well.
# 

# ## Sub-categories of items

# In[19]:


data["Sub-Category"].value_counts()


# In[20]:


# STEFANOS: Disable plotting
# plt.figure(figsize=(15,15))
# plt.pie(data["Sub-Category"].value_counts(), labels= data["Sub-Category"].value_counts().index, autopct ="%2f")
# plt.show()


# #### Here, Sub-Category with highest sale is Binder, follwed by Paper and Furnishings as second and third respectively.

# In[21]:


st_profit=data.groupby(["State"])["Profit"].sum().nlargest(20)


# In[22]:


st_profit


# In[23]:


# STEFANOS: Disable plotting
# plt.figure(figsize=(15,8))
# st_profit.plot.bar()


# In[24]:


# STEFANOS: Disable plotting
# sns.lineplot(data=data, x="Discount", y= "Profit")


# #### So we can see that when discount increases profit decreases

# In[25]:


# STEFANOS: Disable plotting
# data.plot(kind="scatter",x="Sales",y="Profit", c="Discount", colormap="Set1",figsize=(10,10))


# In this scatter plot we can clearly see that more sales does not mean more profit. It depends on discount as well. 
# When Sales is high and there is low discount, the profit margin is higher.

# In[26]:


# %%time
data1= data.groupby("State")[["Sales","Profit"]].sum().sort_values(by="Sales", ascending=False)
# STEFANOS: Disable plotting
# data1[:].plot.bar(color = ["Green","Red"], figsize=(20,12))
# plt.title("Profit-Loss and Sales across States")
# plt.show()


# California and New York generate more profit compared to the other states.

# # Profit-Loss and Sales across Region

# In[27]:


data1= data.groupby("Region")[["Sales","Profit"]].sum().sort_values(by="Sales", ascending=False)
# STEFANOS: Disable plotting
# data1[:].plot.bar(color = ["Blue","Red"], figsize=(10,7))
# plt.title("Profit-Loss and Sales across Region")
# plt.show()


# 
# 
# 
# # Conclusion:
# 
# 
# 1. The western region generates highest profit.
# 2. California, NewYork and Washington generates the most sales compared to the other places. 
# 3. The central region generates lowest profit. 
# 4. Texas, Pennsylvenia, Florida, Illinois, Ohio and some other states are generating loss with high sale. So we need to give some attention towards them.
# 
# Therefore, we have to work more on California and New York. Increase the sales in these states by reducing sales in states like Texas, Florida, Ohio. By decreasing the discount rates in Central region we can increase the profit. Finally we should increase the sale of Office Supplies category as they contribute more.
# 
# 
# 
# 
