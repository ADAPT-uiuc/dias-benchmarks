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


#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# #My Fair Lady "Just You Wait" At the beginning everything sounds nice.
# 
# "Just you wait, Rishi Sunak, just you wait
# 
# You'll be sorry but your tears will be too late
# 
# You'll be broke and I'll have money
# 
# Will I help you, don't be funny
# 
# Just you wait, Rishi Sunak, just you wait
# 
# Just you wait, Rishi Sunak 'till you're sick
# 
# And you screams to fetch a doctor double quick
# 
# I'll be off a second later
# 
# And go straight to the theater
# 
# Oh oh oh, Rishi Sunak, just you wait
# 
# Ooh, Rishi Sunak"
# 
# Lyrics by  Alan Jay Lerner / Frederick Loewe

# In[2]:


import re
import string

import nltk
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# STEFANOS: Unneeded
# from wordcloud import WordCloud
from tqdm.auto import tqdm
import matplotlib.style as style
style.use('fivethirtyeight')


# #Reading Parquet file

# In[3]:


train = pd.read_parquet('input/latest-elected-uk-prime-minister-rishi-sunak/uk_pm.parquet')


# In[4]:


train.head()


# #Or you can have a csv version

# In[5]:


df = pd.read_csv("input/latest-elected-uk-prime-minister-rishi-sunak/uk_pm.csv", delimiter=',', encoding='utf8')
pd.set_option('display.max_columns', None)
df.tail()


# In[6]:


#Code by Leon Wolber https://www.kaggle.com/leonwolber/reddit-nlp-topic-modeling-prediction

print(len(train[train['likecount'] < 500]), 'tweets with less than 500 dislikes')
print(len(train[train['likecount'] > 500]), 'tweets with more than 500 dislikes')


# #Tweet with more comments???? I'm making my Wine Reviews (aka drinking)

# In[7]:


#Code by Leon Wolber https://www.kaggle.com/leonwolber/reddit-nlp-topic-modeling-prediction

# video with the most comments

train[train['likecount'] == train['likecount'].max()]['text'].iloc[0]


# #We don't have DISLIKES yet? Just wait Prime Minister! 

# In[8]:


#Code by Leon Wolber https://www.kaggle.com/leonwolber/reddit-nlp-topic-modeling-prediction

def remove_line_breaks(text):
    text = text.replace('\r', ' ').replace('\n', ' ')
    return text

#remove punctuation
def remove_punctuation(text):
    re_replacements = re.compile("__[A-Z]+__")  # such as __NAME__, __LINK__
    re_punctuation = re.compile("[%s]" % re.escape(string.punctuation))
    '''Escape all the characters in pattern except ASCII letters and numbers'''
    tokens = word_tokenize(text)
    tokens_zero_punctuation = []
    for token in tokens:
        if not re_replacements.match(token):
            token = re_punctuation.sub(" ", token)
        tokens_zero_punctuation.append(token)
    return ' '.join(tokens_zero_punctuation)

def remove_special_characters(text):
    text = re.sub('[^a-zA-z0-9\s]', '', text)
    return text

def lowercase(text):
    text_low = [token.lower() for token in word_tokenize(text)]
    return ' '.join(text_low)

def remove_stopwords(text):
    stop = set(stopwords.words('english'))
    word_tokens = nltk.word_tokenize(text)
    text = " ".join([word for word in word_tokens if word not in stop])
    return text

#remobe one character words
def remove_one_character_words(text):
    '''Remove words from dataset that contain only 1 character'''
    text_high_use = [token for token in word_tokenize(text) if len(token)>1]      
    return ' '.join(text_high_use)   
    
#%%
# Stemming with 'Snowball stemmer" package
def stem(text):
    stemmer = nltk.stem.snowball.SnowballStemmer('english')
    text_stemmed = [stemmer.stem(token) for token in word_tokenize(text)]        
    return ' '.join(text_stemmed)

def lemma(text):
    wordnet_lemmatizer = WordNetLemmatizer()
    word_tokens = nltk.word_tokenize(text)
    text_lemma = " ".join([wordnet_lemmatizer.lemmatize(word) for word in word_tokens])       
    return ' '.join(text_lemma)


#break sentences to individual word list
def sentence_word(text):
    word_tokens = nltk.word_tokenize(text)
    return word_tokens
#break paragraphs to sentence token 
def paragraph_sentence(text):
    sent_token = nltk.sent_tokenize(text)
    return sent_token    


def tokenize(text):
    """Return a list of words in a text."""
    return re.findall(r'\w+', text)

def remove_numbers(text):
    no_nums = re.sub(r'\d+', '', text)
    return ''.join(no_nums)



def clean_text(text):
    _steps = [
    remove_line_breaks,
    remove_one_character_words,
    remove_special_characters,
    lowercase,
    remove_punctuation,
    remove_stopwords,
    stem,
    remove_numbers
]
    for step in _steps:
        text=step(text)
    return text   
#%%


# In[9]:


#https://stackoverflow.com/questions/55557004/getting-attributeerror-float-object-has-no-attribute-replace-error-while
#To avoid with tqdm AttributeError: 'float' object has no attribute

train["text"] = train["text"].astype(str)
train["text"] = [x.replace(':',' ') for x in train["text"]]


# In[10]:


train['clean_text'] = pd.Series([clean_text(i) for i in tqdm(train['text'])])


# In[11]:


words = train["clean_text"].values


# In[12]:


#Code by Leon Wolber https://www.kaggle.com/leonwolber/reddit-nlp-topic-modeling-prediction

ls = []

for i in words:
    ls.append(str(i))


# In[13]:


ls[:5]


# In[14]:


# STEFANOS: Disable plotting
# #Code by Leon Wolber https://www.kaggle.com/leonwolber/reddit-nlp-topic-modeling-prediction

# # The wordcloud 
# plt.figure(figsize=(16,13))
# wc = WordCloud(background_color="lightblue", colormap='Set2', max_words=1000, max_font_size= 200,  width=1600, height=800)
# wc.generate(" ".join(ls))
# plt.title("Most discussed terms", fontsize=20)
# plt.imshow(wc.recolor( colormap= 'Set2' , random_state=17), alpha=0.98, interpolation="bilinear", )
# plt.axis('off')


# In[15]:


#Code by Leon Wolber https://www.kaggle.com/leonwolber/reddit-nlp-topic-modeling-prediction

most_pop = train.sort_values('likecount', ascending =False)[['text', 'likecount']].head(12)

most_pop['target1'] = most_pop['likecount']/1000


# In[16]:


# STEFANOS: Disable plotting

# #Code by Leon Wolber https://www.kaggle.com/leonwolber/reddit-nlp-topic-modeling-prediction

# plt.figure(figsize = (30,35))

# sns.barplot(data = most_pop, y = 'text', x = 'target1', color = 'c')
# plt.xticks(fontsize=50, rotation=0)
# plt.yticks(fontsize=50, rotation=0)
# plt.xlabel('Votes in Thousands', fontsize = 21)
# plt.ylabel('')
# plt.title('Most popular tweets', fontsize = 50);


# #I can't even read anything. Just: "Congratulations Rishi Sunak." Just Wait PM Sunak.

# In[17]:


import collections.abc
#gensim aliases to be done manually.
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)
import nltk


# In[18]:


stemmer = SnowballStemmer('english')


# In[19]:


# STEFANOS: Disable. Connects to net.
# nltk.download('wordnet')


# In[20]:


#Code by Leon Wolber https://www.kaggle.com/leonwolber/reddit-nlp-topic-modeling-prediction

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


# #At the beginning, everything is so nice

# In[21]:


train['text'].iloc[2]


# #I ran the snippet below to avoid error on the next snippet (after that one)

# In[22]:


import nltk
# STEFANOS: Disable. Connects to net.
# nltk.download('omw-1.4')


# In[23]:


#Code by Leon Wolber https://www.kaggle.com/leonwolber/reddit-nlp-topic-modeling-prediction

doc_sample = train['text'].iloc[1]
print('original document: ')

words = []

for word in doc_sample.split(' '):
    words.append(word)
    
    
print(words)
print('\n\n tokenized and lemmatized document: ')
print(preprocess(doc_sample))


# In[24]:


train['text'] = train['text'].astype(str)


# #Above it was suppose to be "clean_text"  Though I got an error.

# In[25]:


#Code by Leon Wolber https://www.kaggle.com/leonwolber/reddit-nlp-topic-modeling-prediction

words = []

for i in train['text']:
        words.append(i.split(' '))


# In[26]:


#Code by Leon Wolber https://www.kaggle.com/leonwolber/reddit-nlp-topic-modeling-prediction

dictionary = gensim.corpora.Dictionary(words)

count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break


# In[27]:


bow_corpus = [dictionary.doc2bow(doc) for doc in words]
bow_corpus[4310]


# In[28]:


#Code by Leon Wolber https://www.kaggle.com/leonwolber/reddit-nlp-topic-modeling-prediction

bow_doc_4310 = bow_corpus[4310]

for i in range(len(bow_doc_4310)):
    print("Word {} (\"{}\") appears {} time.".format(bow_doc_4310[i][0], 
                                               dictionary[bow_doc_4310[i][0]], 
bow_doc_4310[i][1]))


# In[29]:


# STEFANOS: Disable the rest. It's ML.

