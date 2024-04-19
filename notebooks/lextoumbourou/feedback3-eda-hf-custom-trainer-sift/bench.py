#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[1]:


import numpy as np
import os
import pandas as pd


# # Load Data

# In[2]:


train_df = pd.read_csv('input/feedback-prize-english-language-learning/train.csv')
test_df = pd.read_csv('input/feedback-prize-english-language-learning/test.csv')


# # -- STEFANOS -- Replicate Data

# In[3]:


factor = 500
if "IREWR_LESS_REPLICATION" in os.environ and os.environ["IREWR_LESS_REPLICATION"] == "True":
    factor = factor//5
train_df = pd.concat([train_df]*factor)
test_df = pd.concat([test_df]*factor)
# train_df.info()


# Let's see a row from each dataset.

# In[4]:


train_df.head()


# In[5]:


test_df.head()


# Then the size of each dataset.

# In[6]:


len(train_df), len(test_df)


# In[7]:


LABEL_COLUMNS = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']


# # Text Examples

# ## Random Examples

# In[8]:


texts = train_df.sample(frac=1, random_state=420).head(4)


# ## Lowest Scoring Examples

# In[9]:


train_df['total_score'] = train_df[LABEL_COLUMNS].sum(axis=1)
lowest_df = train_df.sort_values('total_score').head(4)


# ## Highest Scoring Examples

# In[10]:


train_df['total_score'] = train_df[LABEL_COLUMNS].sum(axis=1)
highest_df = train_df.sort_values('total_score', ascending=False).head(4)


# # Text Overview

# ## Word Count

# In[11]:


train_df['word_count'] = train_df.full_text.apply(lambda x: len(x.split()))


# Mean word count:

# In[12]:


train_df['word_count'].mean()


# Max word count:

# In[13]:


train_df['word_count'].max()

