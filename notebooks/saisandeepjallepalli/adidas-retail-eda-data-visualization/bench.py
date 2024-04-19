#!/usr/bin/env python
# coding: utf-8

# <a id="1"></a>
# # <p style="padding:10px;background-color:#A23434;margin:0;color:#CDCA34;font-family:newtimeroman;font-size:100%;text-align:center;border-radius: 15px 50px;overflow:hidden;font-weight:500">Importing Libraries</p>

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
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# <a id="1"></a>
# # <p style="padding:10px;background-color:#A23434;margin:0;color:#CDCA34;font-family:newtimeroman;font-size:100%;text-align:center;border-radius: 15px 50px;overflow:hidden;font-weight:500">Reading Dataset</p>

# In[2]:


df = pd.read_csv("input/adidas-us-retail-products-dataset/adidas.csv")


# # -- STEFANOS -- Replicate Data

# In[3]:


factor = 1000
df = pd.concat([df]*factor)
# df.info()


# <a id="1"></a>
# # <p style="padding:10px;background-color:#A23434;margin:0;color:#CDCA34;font-family:newtimeroman;font-size:100%;text-align:center;border-radius: 15px 50px;overflow:hidden;font-weight:500">Basic Information</p>

# In[4]:


# First 5 records in the DataFrame

df.head()


# In[5]:


# Checking for Null Values in the DataFrame

df.isna().sum()

# There 16 Null Values for Original_price Column


# In[6]:


# Shape of the DataFrame

df.shape


# In[7]:


# Information about the DataFrame

# df.info()


# <a id="1"></a>
# # <p style="padding:10px;background-color:#A23434;margin:0;color:#CDCA34;font-family:newtimeroman;font-size:100%;text-align:center;border-radius: 15px 50px;overflow:hidden;font-weight:500">Statistical Information</p>

# In[8]:


# Correlation Between Columns

df.corr()


# In[9]:


# Statistical informtion about the DataFrame

# df.describe().T


# <a id="1"></a>
# # <p style="padding:10px;background-color:#A23434;margin:0;color:#CDCA34;font-family:newtimeroman;font-size:100%;text-align:center;border-radius: 15px 50px;overflow:hidden;font-weight:500">Feature Engineering</p>

# In[10]:


# The Percentage of Missing Values in the 'original_price' Column

100 * (df['original_price'].isna().sum() / len(df))

# As the percntage of Null Records is less than 5%, hence dropping the Null record Rows


# In[11]:


# Dropping Null Values in the DataFrame

df.dropna(inplace=True, axis=0)


# In[12]:


# Checking for Null Values in the DataFrame

df.isna().sum()


# In[13]:


# Dropping 'currency' column as all records have 'USD' as currency
# Dropping 'source' column as all records have 'adidas United States' as value
# Dropping 'brand', 'country', 'language' columns as all records have same value

df.drop([ 'brand', 'country', 'language', 'source_website', 'images', 'crawled_at', 'url', 'sku', 'currency','source', 'description'], axis=1, inplace=True)


# In[14]:


# First 5 records in the DataFrame

df.head()


# In[15]:


# Removing '$' from the DataFrame

df['original_price'] = df['original_price'].str.split('$')
df['original_price'] = df['original_price'].str[1]


# In[16]:


# First 5 records in the DataFrame

df.head()


# In[17]:


# Shape of the DataFrame

df.shape


# In[18]:


# Checking the Data Types for the columns

df.dtypes


# In[19]:


# Creating 'Category' Column

df['Category'] = df['breadcrumbs'].str.split("/")
df['Category'] = df['Category'].str[0]


# In[20]:


# Creating 'Product Type' Column

df['Product_Type'] = df['breadcrumbs'].str.split("/")
df['Product_Type'] = df['Product_Type'].str[1]


# In[21]:


# Droping 'breadcrumbs' Column

df.drop(['breadcrumbs', 'category'], axis=1, inplace=True)


# In[22]:


# Changing the DataType of 'original_price' from object to int64

df['original_price'] = df['original_price'].astype('int64')


# In[23]:


# Checking the Data Types for the columns

df.dtypes


# <a id="1"></a>
# # <p style="padding:10px;background-color:#A23434;margin:0;color:#CDCA34;font-family:newtimeroman;font-size:100%;text-align:center;border-radius: 15px 50px;overflow:hidden;font-weight:500">EDA and Data Visualization</p>

# # Heat-Map

# In[24]:


# STEFANOS: Disable plotting
# sns.heatmap(df.corr(), annot=True, cmap='magma')
# plt.show();


# # Distribution Plots

# In[25]:


# Distribution Plot for selling price

# STEFANOS: Disable plotting
# sns.displot(df['selling_price'], kde=True);

# The Below Graph is 'Right Skewed' with Majority of Data falling between 10-70.


# In[26]:


# Distribution Plot for origial price

# STEFANOS: Disable plotting
# sns.displot(df['original_price'], kde=True);

# The Below Graph is 'Right Skewed' with Majority of Data falling between 10-90.


# In[27]:


# Distribution Plot for average rating

# STEFANOS: Disable plotting
# sns.displot(df['average_rating'], kde=True);

# The Below Graph is 'Left Skewed' with Majority of Data falling between 4.2-5.0.


# In[28]:


# Distribution Plot for reviews count

# STEFANOS: Disable plotting
# sns.displot(df['reviews_count'], kde=True);

# The Below Graph is 'Right Skewed' with Majority of Data falling below 600.


# # Product Name

# In[29]:


# No.of Unique Products in the DataFrame

df['name'].nunique()


# In[30]:


# Top 10 - Products sold in the DataFrame

df['name'].value_counts()[:10]


# In[31]:


# Top 5 - Products sold in the DataFrame


# STEFANOS: Disable plotting
# plt.figure(figsize=(12,4))
# df['name'].value_counts()[:5].plot(kind='barh', color={'#864879','#2D4263','#C84B31', '#ECDBBA', '#B3541E'})
# plt.ylabel("Product Name")
# plt.xlabel("No.of Units Sold")
# plt.title("Product Name Vs. No.of Units Sold")
# plt.show();
df['name'].value_counts()[:5]


# #### The Product '**ZX 1K Boost Shoes**' is the most sold Product/Recorded in the DataFrame with '**24**' units.

# # Selling Price

# In[32]:


# No.of Unique selling price in the DataFrame

df['selling_price'].nunique()


# In[33]:


# Top 15 - Products interms of similar selling price in the DataFrame

df['selling_price'].value_counts()[:15]


# In[34]:


# Top 5 - Products interms of similar selling price in the DataFrame

# STEFANOS: Disable plotting
# plt.figure(figsize=(12,4))
# df['selling_price'].value_counts()[:5].plot(kind='barh', color={'#864879','#2D4263','#C84B31', '#ECDBBA', '#B3541E'})
# plt.ylabel("Product Selling Price ($USD)")
# plt.xlabel("No.of Products")
# plt.title("Product Selling Price ($USD) Vs. No.of Products")
# plt.show();
df['selling_price'].value_counts()[:5]


# In[35]:


# Top 15 - Products interms of similar selling price in the DataFrame

# STEFANOS: Disable plotting
# labels = df['selling_price'].value_counts().head(15).index
# pie, ax = plt.subplots(figsize=[20,8])
# plt.pie(x=df['selling_price'].value_counts().head(15).values, autopct="%.1f%%", explode=[0.06]*15, labels=labels, pctdistance=0.5)
# plt.title("Top 15 - Products interms of similar selling price ($USD)", fontsize=14);


# ## Maximum Selling Price

# In[36]:


# Maximum Selling Price

print("The Maximun Selling Price is:",df['selling_price'].max(),"USD")


# ## Minimum Selling Price

# In[37]:


# Minimum Selling Price

print("The Minimum Selling Price is:",df['selling_price'].min(),"USD")


# ## Average Selling Price

# In[38]:


# Average Selling Price

print("The Average Selling Price is:",round(df['selling_price'].mean(),2),"USD")


# In[39]:


# Top 10 - Highest Selling Prices in USD

set(df['selling_price'].sort_values(ascending=False)[:30].values)


# In[40]:


# Top 10 - Least Selling Prices in USD

set(df['selling_price'].sort_values()[:40].values)


# #### Highest Selling price is '**240 USD**' and Least is '**9 USD**' and product(s) with selling price of '**56 USD**' is sold '**54**' times.

# # Original Price

# In[41]:


# No.of Unique original price in the DataFrame

df['original_price'].nunique()


# In[42]:


# Top 15 - Products interms of similar original price in the DataFrame

df['original_price'].value_counts()[:15]


# In[43]:


# Top 5 - Products interms of similar origial price in the DataFrame

# STEFANOS: Disable plotting
# plt.figure(figsize=(12,4))
# df['original_price'].value_counts()[:5].plot(kind='barh', color={'#864879','#2D4263','#C84B31', '#ECDBBA', '#B3541E'})
# plt.ylabel("Original Product Price ($USD)")
# plt.xlabel("No.of Products")
# plt.title("Original Product Price ($USD) Vs. No.of Products")
# plt.show();
df['original_price'].value_counts()[:5]


# In[44]:


# Top 15 - Products interms of similar Original price in the DataFrame

# STEFANOS: Disable plotting
# labels = df['original_price'].value_counts().head(15).index
# pie, ax = plt.subplots(figsize=[20,8])
# plt.pie(x=df['original_price'].value_counts().head(15).values, autopct="%.1f%%", explode=[0.06]*15, labels=labels, pctdistance=0.5)
# plt.title("Top 15 - Products interms of similar original price ($USD)", fontsize=14);


# ## Maximum Original Price

# In[45]:


# Maximum Original Price

print("The Maximun Original Price is:",df['original_price'].max(),"USD")


# ## Minimum Original Price

# In[46]:


# Minimum Original Price

print("The Minimum Original Price is:",df['original_price'].min(),"USD")


# ## Average Original Price

# In[47]:


# Average Original Price

print("The Average Original Price is:",round(df['original_price'].mean(),2),"USD")


# In[48]:


# Top 10 - Highest Original Prices in USD

set(df['original_price'].sort_values(ascending=False)[:80].values)


# In[49]:


# Top 10 - Least Original Prices in USD

set(df['original_price'].sort_values().values[:100])


# #### Highest Original price is '**300 USD**' and Least is '**14 USD**' and product(s) with selling price of '**65 USD**' is sold '**68**' times.

# # Discount

# In[50]:


# Calculating Discount

df['Discount'] = df['original_price'] - df['selling_price']


# In[51]:


# No.of Unique Discount in the DataFrame

df['Discount'].nunique()


# In[52]:


# Top 15 - Highest Discount Amount in the DataFrame

list(set(df['Discount'].unique()))[-15::]


# In[53]:


# Top 15 - Least Discount Amount in the DataFrame

list(set(df['Discount'].unique()))[:15]


# In[54]:


# Top 15 - Most Given Discount Amount in the DataFrame

df['Discount'].value_counts()[:15]


# In[55]:


# Top 5 - Most Given Discount Amount in the DataFrame

# STEFANOS: Disable plotting
# plt.figure(figsize=(12,4))
# df['Discount'].value_counts()[:5].plot(kind='barh', color={'#864879','#2D4263','#C84B31', '#ECDBBA', '#B3541E'})
# plt.ylabel("Discount Amount")
# plt.xlabel("No.of times Particular Disount Amount Given")
# plt.title("Discount Amount Vs. No.of times Particular Disount Amount Given")
# plt.show();
df['Discount'].value_counts()[:5]


# In[56]:


# Top 15 - Most Given Discount Amount in the DataFrame

labels = df['Discount'].value_counts().head(15).index
# STEFANOS: Disable plotting
# pie, ax = plt.subplots(figsize=[20,8])
# plt.pie(x=df['Discount'].value_counts().head(15).values, autopct="%.1f%%", explode=[0.06]*15, labels=labels, pctdistance=0.5)
# plt.title("Top 15 - Most Given Discount Amount in the DataFrame", fontsize=14);
_ = df['Discount'].value_counts().head(15).values


# #### Highest Discount Amount is '**84 USD**' and Least is '**2 USD**' and product(s) with Disount Amount of '**5 USD**' is sold '**73**' times.

# # Discount Percentage(%)

# In[57]:


# Calculating Discount percentage(%)

df['Discount(%)'] = round(((df['original_price'] - df['selling_price']) / (df['original_price']))*100,2)


# In[58]:


# No.of Unique Discount Percentage(%) in the DataFrame

df['Discount(%)'].nunique()


# In[59]:


# Top 15 - Highest Discount Percentage(%) in the DataFrame

top_discount_percnet = list(set(df['Discount(%)'].unique()))
top_discount_percnet.sort(reverse=True)
print(top_discount_percnet[:15])


# In[60]:


# Top 15 - Least Discount Percentage(%) in the DataFrame

least_discount_percnet = list(set(df['Discount(%)'].unique()))
least_discount_percnet.sort(reverse=False)
print(least_discount_percnet[:15])


# In[61]:


# Top 15 - Most Given Discount Percentage(%) in the DataFrame

df['Discount(%)'].value_counts().head(15)


# In[62]:


# Top 5 - Most Given Discount Percentage in the DataFrame

# STEFANOS: Disable plotting
# plt.figure(figsize=(12,4))
# df['Discount(%)'].value_counts().head(5).plot(kind='barh', color={'#864879','#2D4263','#C84B31', '#ECDBBA', '#B3541E'})
# plt.ylabel("Discount Percentage")
# plt.xlabel("No.of times Particular Disount Percentage Given")
# plt.title("Discount Percentage Vs. No.of times Particular Disount Percentage Given")
# plt.show();
df['Discount(%)'].value_counts().head(5)


# In[63]:


# Top 15 - Most Given Discount Percentage in the DataFrame

labels = df['Discount(%)'].value_counts().head(15).index
# STEFANOS: Disable plotting
# pie, ax = plt.subplots(figsize=[20,15])
# plt.pie(x=df['Discount(%)'].value_counts().head(15).values, autopct="%.1f%%", explode=[0.06]*15, labels=labels, pctdistance=0.5)
# plt.title("Top 15 - Most Given Discount Percenatage in the DataFrame", fontsize=14);
# plt.tight_layout();
_=df['Discount(%)'].value_counts().head(15).values


# #### Highest Discount% is '**50%**' and Least is '**7.14%**' and product(s) with Disount Percentage of '**20**' is sold '**254**' times.

# # Color

# In[64]:


# Unique colors in the DataFrame

df['color'].unique()


# In[65]:


# No.of Unique colors in the DataFrame

df['color'].nunique()


# In[66]:


# No.of Products for each Color

df['color'].value_counts()


# In[67]:


# No.of Products for each Color

labels = df['color'].value_counts().index
# STEFANOS: Disable plotting
# pie, ax = plt.subplots(figsize=[20,15])
# plt.pie(x=df['color'].value_counts().values, autopct="%.1f%%", explode=[0.03]*18, labels=labels, pctdistance=0.5)
# plt.title("No.of Products for each Color in the DataFrame", fontsize=14);
# plt.tight_layout();
_ = x=df['color'].value_counts().values


# # Which Color is most popular among genders?

# In[68]:


# Popular Color in Kids Category

df.groupby(['Category','color']).size()['Kids']


# In[69]:


# Popular Color in Womens Category

df.groupby(['Category','color']).size()['Women']


# In[70]:


# Popular Color in Mens Category

df.groupby(['Category','color']).size()['Men']


# #### The top color in **Mens** category is '**White**' with '67' sales,the top color in **Womens** category is '**White**' with '92' sales and the top color in **Kids** category is '**White**' with '26' sales

# # Availability

# In[71]:


# Unique Availability values in the DataFrame

df['availability'].unique()


# In[72]:


# No.of Unique Availability values in the DataFrame

df['availability'].nunique()


# In[73]:


# No.of Products according to Avaialability

df['availability'].value_counts()


# #### There are '**826**' products which are '**InStock**' and '**3**' are '**OutOfStock**'.

# ## Product Type

# In[74]:


# Unique Product Type values in the DataFrame

df['Product_Type'].unique()


# In[75]:


# No.of Unique Product Type values in the DataFrame

df['Product_Type'].nunique()


# In[76]:


# No.of Products according to Product Type

df['Product_Type'].value_counts()


# In[77]:


# No.of Products according to Product Type

# STEFANOS: Disable plotting
# plt.figure(figsize=(12,4))
# df['Product_Type'].value_counts().plot(kind='barh', color={'#864879','#2D4263','#C84B31'})
# plt.ylabel("Product Type")
# plt.xlabel("No.of Products Sold According to Particular - Product Type")
# plt.title("Product Type Vs. No.of Products Sold According to Particular - Product Type")
# plt.show();
df['Product_Type'].value_counts()


# In[78]:


# No.of Products according to Product Type

labels = df['Product_Type'].value_counts().index
# STEFANOS: Disable plotting
# pie, ax = plt.subplots(figsize=[10,4])
# plt.pie(x=df['Product_Type'].value_counts().values, autopct="%.1f%%", explode=[0.05]*3, labels=labels, pctdistance=0.5)
# plt.title("No.of Products according to Product Type in the DataFrame", fontsize=14);
# plt.tight_layout();
_ = df['Product_Type'].value_counts().values


# ## Maximum Selling Price for Each Product Type

# In[79]:


df.groupby(['Product_Type', 'Category']).max()['selling_price']


# ## Minimum Selling Price for Each Product Type

# In[80]:


df.groupby(['Product_Type', 'Category']).min()['selling_price']


# ## Average Selling Price for Each Product Type

# In[81]:


df.groupby(['Product_Type', 'Category']).mean()['selling_price']


# #### The Product Type with most sales is '**Shoes**', and Maximum Selling Price of '**240$**' according to Selling Price. 

# # Category

# In[82]:


# Unique Category values in the DataFrame

df['Category'].unique()


# In[83]:


# No.of Unique Category values in the DataFrame

df['Category'].nunique()


# In[84]:


# No.of Products according to Category

df['Category'].value_counts()


# In[85]:


# Top-5 Products according to Category

# STEFANOS: Disable plotting
# plt.figure(figsize=(12,4))
# df['Category'].value_counts().head(5).plot(kind='barh', color={'#864879','#2D4263','#C84B31', '#ECDBBA', '#B3541E'})
# plt.ylabel("Category")
# plt.xlabel("No.of Products Sold According to Particular - Category")
# plt.title("Top 5 - Products Sold according to Category in the DataFrame")
# plt.show();
df['Category'].value_counts().head(5)


# In[86]:


# Top-5 Products according to Category

labels = df['Category'].value_counts().head(5).index
# STEFANOS: Disable plotting
# pie, ax = plt.subplots(figsize=[10,4])
# plt.pie(x=df['Category'].value_counts().head(5).values, autopct="%.1f%%", explode=[0.05]*5, labels=labels, pctdistance=0.5)
# plt.title("Top 5 - Products Sold according to Category in the DataFrame", fontsize=14);
# plt.tight_layout();
_ = df['Category'].value_counts().head(5).values


# #### The 'Category' with highest no.of Products Sold is '**Women**' with '**342 Products**', followed by 'Men' and 'Women'.

# # Average Rating

# In[87]:


# Unique 'Average Rating'in the DataFrame

df['average_rating'].unique()


# In[88]:


# No.of Unique 'Average Rating'in the DataFrame

df['average_rating'].nunique()


# In[89]:


# No.of time particular Average Rating Provided

df['average_rating'].value_counts()


# In[90]:


# Top 5 - Average Rating Provided

# STEFANOS: Disable plotting
# plt.figure(figsize=(12,4))
# df['average_rating'].value_counts().head(5).plot(kind='barh', color={'#864879','#2D4263','#C84B31', '#ECDBBA', '#B3541E'})
# plt.ylabel("Average Rating")
# plt.xlabel("No.of times particular rating provided")
# plt.title("Top 5 - Average Rating Provided in the DataFrame")
# plt.show();
df['average_rating'].value_counts().head(5)


# In[91]:


# Top 5 - Average Rating Provided

labels = df['average_rating'].value_counts().head(5).index
# STEFANOS: Disable plotting
# pie, ax = plt.subplots(figsize=[10,4])
# plt.pie(x=df['average_rating'].value_counts().head(5).values, autopct="%.1f%%", explode=[0.05]*5, labels=labels, pctdistance=0.5)
# plt.title("# Top 5 - Average Rating Provided in the DataFrame", fontsize=14);
# plt.tight_layout();
_ = df['average_rating'].value_counts().head(5).values


# In[92]:


# Maxmimum Average Rating

df['average_rating'].max()


# In[93]:


# Minimum Average Rating

df['average_rating'].min()


# In[94]:


# Mean Average Rating

round(df['average_rating'].mean(),2)


# In[95]:


# Maximum Average Rating across 'Product Type' and 'Category'

df.groupby(['Product_Type', 'Category']).max()['average_rating']


# In[96]:


# Minimum Average Rating across 'Product Type' and 'Category'

df.groupby(['Product_Type', 'Category']).min()['average_rating']


# #### The **Maximum** Average Rating among all products is '**5.0**' and **Minimum** Average Rating among all products is '**1.0**'

# # Reviews Count

# In[97]:


# No.of Unique 'Reviews Count' in the DataFrame

df['reviews_count'].nunique()


# In[98]:


# Top 10 - 'Reviews Count' and no.of particular 'Reviews Count' occurances

df['reviews_count'].value_counts().head(10)


# In[99]:


# Top 5 - 'Reviews Count' and no.of particular 'Reviews Count' occurances

# STEFANOS: Disable plotting
# plt.figure(figsize=(12,4))
# df['average_rating'].value_counts().head(5).plot(kind='barh', color={'#864879','#2D4263','#C84B31', '#ECDBBA', '#B3541E'})
# plt.ylabel('Reviews Count')
# plt.xlabel('no.of particular Reviews Count occurances')
# plt.title("Top 5 - 'Reviews Count' and no.of particular 'Reviews Count' occurances")
# plt.show();
df['average_rating'].value_counts().head(5)


# In[100]:


# Maxmimum 'Reviews Count'

df['reviews_count'].max()


# In[101]:


# Minimum 'Reviews Count'

df['reviews_count'].min()


# In[102]:


# Mean 'Reviews Count'

round(df['reviews_count'].mean(),2)


# In[103]:


# Maximum 'Reviews Count' across 'Product Type' and 'Category'

df.groupby(['Product_Type', 'Category']).max()['reviews_count']


# In[104]:


# Minimum Reviews Count' across 'Product Type' and 'Category'

df.groupby(['Product_Type', 'Category']).min()['reviews_count']


# #### The **Maximum** 'Reviews Count' among all products is '**11750**' and **Minimum** 'Reviews Count' among all products is '**1**'

# # Reviews Count Vs. Average Rating

# In[105]:


# STEFANOS: Disable plotting
# sns.scatterplot(data=df, x='average_rating', y='reviews_count')


# #### The Correlation Between Reviews Count & Average Rating is '**0.024**' (positive correlation), from the graph we may can conclude that there are more number of reviews when particular product rated above **3.5** 

# ## End of the Notebook, Thanks for Watching, "**Do Upvote**" if it is helpful.
