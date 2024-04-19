#!/usr/bin/env python
# coding: utf-8

# # Feedback Prize - Evaluating Student Writing
# 
# Georgia State University (GSU) is an undergraduate and graduate urban public research institution in Atlanta. U.S. News & World Report ranked GSU as one of the most innovative universities in the nation. GSU awards more bachelor’s degrees to African-Americans than any other non-profit college or university in the country. GSU and The Learning Agency Lab, an independent nonprofit based in Arizona, are focused on developing science of learning-based tools and programs for social good.
# 
# In this competition, you’ll identify elements in student writing. More specifically, you will automatically segment texts and classify argumentative and rhetorical elements in essays written by 6th-12th grade students. You'll have access to the largest dataset of student writing ever released in order to test your skills in natural language processing, a fast-growing area of data science.
# 
# If successful, you'll make it easier for students to receive feedback on their writing and increase opportunities to improve writing outcomes. Virtual writing tutors and automated writing systems can leverage these algorithms while teachers may use them to reduce grading time. The open-sourced algorithms you come up with will allow any educational organization to better help young writers develop.

# In[1]:


import numpy as np
import os
# STEFANOS: Conditionally import Modin Pandas

import pandas as pd
from glob import glob
# import matplotlib.pyplot as plt
# # STEFANOS: Disable, can't be parsed. Doesn't change the behavior especially since we've disabled plotting.
# # %matplotlib inline
# import matplotlib.style as style
# style.use('fivethirtyeight')

from nltk.corpus import stopwords

import warnings
warnings.filterwarnings('ignore')
# STEFANOS: Remove unneeded imports.
# import spacy
from sklearn.feature_extraction.text import CountVectorizer
import os


# In[2]:


train = pd.read_csv('input/feedback-prize-2021/train.csv')
if "IREWR_LESS_REPLICATION" in os.environ and os.environ["IREWR_LESS_REPLICATION"] == "True":
    train = train[:5000]

train[['discourse_id', 'discourse_start', 'discourse_end']] = train[['discourse_id', 'discourse_start', 'discourse_end']].astype(int)

sample_submission = pd.read_csv('input/feedback-prize-2021/sample_submission.csv')

#The glob module finds all the pathnames matching a specified pattern according to the rules used by the Unix shell
train_txt = glob('input/feedback-prize-2021/train/*.txt')
test_txt = glob('input/feedback-prize-2021/test/*.txt')


# # Introduction to the competition
# 
# Basically, we have a bunch of essays written by kids in the age range of about 12-18 years old in which we have to find word sequences that can be classified as one of 7 "discourse types". These are:
# 
# - Lead - an introduction that begins with a statistic, a quotation, a description, or some other device to grab the reader’s attention and point toward the thesis
# - Position - an opinion or conclusion on the main question
# - Claim - a claim that supports the position
# - Counterclaim - a claim that refutes another claim or gives an opposing reason to the position
# - Rebuttal - a claim that refutes a counterclaim
# - Evidence - ideas or examples that support claims, counterclaims, or rebuttals.
# - Concluding Statement - a concluding statement that restates the claims
# 
# Let's look at the full text of one essay first.

# In[3]:


# Disable shell code.
# !cat input/feedback-prize-2021/train/423A1CA112E2.txt


# The train dataset gives us the following human annotations that are extracted from this essay.

# In[4]:


train.query('id == "423A1CA112E2"')


# Kaggle gives us the following field descriptions:
# - id - ID code for essay response
# - discourse_id - ID code for discourse element
# - discourse_start - character position where discourse element begins in the essay response
# - discourse_end - character position where discourse element ends in the essay response
# - discourse_text - text of discourse element
# - discourse_type - classification of discourse element
# - discourse_type_num - enumerated class label of discourse element
# - predictionstring - the word indices of the training sample, as required for predictions
# 
# The Ground Truth here is a combination of the discourse type and the prediction string. The predictionstring corresponds to the index of the words in the essay and the predicted discourse type for this sequence of words should be correct. There can be partial matches, if the correct discourse type is predicted but on a longer or shorter sequence of words than specified in the Ground Truth.
# 
# As we can see, not necessarily all text of an essay is part of a discourse. In this case, the title is not part of any discourse.
# 
# 
# # Lenght of the discourse_text and predictionstring
# First, I would like to check if the discourse_text and the predictionstring always have the same number of words (as they should).

# In[5]:


#add columns
train["discourse_len"] = train["discourse_text"].apply(lambda x: len(x.split()))
train["pred_len"] = train["predictionstring"].apply(lambda x: len(x.split()))


cols_to_display = ['discourse_id', 'discourse_text', 'discourse_type','predictionstring', 'discourse_len', 'pred_len']
train[cols_to_display].head()


# Is this always correct? No, I find 468 discourses where this goes wrong (by one word)

# In[6]:


print(f"The total number of discourses is {len(train)}")
train.query('discourse_len != pred_len')[cols_to_display]


# Let's check the first one.

# In[7]:


print(train.query('discourse_id == 1622473475289')['discourse_text'].values[0])
print(train.query('discourse_id == 1622473475289')['discourse_text'].values[0].split())
print(len(train.query('discourse_id == 1622473475289')['discourse_text'].values[0].split()))


# The length of 19 words seems correct to me, and the length of the predictionstring also really seems to be 18. Something to keep in mind.
# 
# **Update:** the answers to this can be found in discussion topic: [Mystery Solved - Discrepancy Between PredictionString and DiscourseText](https://www.kaggle.com/c/feedback-prize-2021/discussion/297591)

# In[8]:


print(train.query('discourse_id == 1622473475289')['predictionstring'].values[0])
print(train.query('discourse_id == 1622473475289')['predictionstring'].values[0].split())
print(len(train.query('discourse_id == 1622473475289')['predictionstring'].values[0].split()))


# # Length and frequency and relative position per discourse_type
# 
# Is there a correlation between the length of a discourse and the class (discourse_type)? Yes, there is. Evidence is the longest discount type on average. When looking at the frequencies of occurence, we see that Counterclaim and Rebuttal are relatively rare

# In[9]:


# STEFANOS: Disable plotting
# fig = plt.figure(figsize=(12,8))

# ax1 = fig.add_subplot(211)
# ax1 = train.groupby('discourse_type')['discourse_len'].mean().sort_values().plot(kind="barh")
ax1 = train.groupby('discourse_type')['discourse_len'].mean().sort_values()
# ax1.set_title("Average number of words versus Discourse Type", fontsize=14, fontweight = 'bold')
# ax1.set_xlabel("Average number of words", fontsize = 10)
# ax1.set_ylabel("")

# ax2 = fig.add_subplot(212)
# ax2 = train.groupby('discourse_type')['discourse_type'].count().sort_values().plot(kind="barh")
ax2 = train.groupby('discourse_type')['discourse_type'].count().sort_values()
# ax2.get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ','))) #add thousands separator
# ax2.set_title("Frequency of Discourse Type in all essays", fontsize=14, fontweight = 'bold')
# ax2.set_xlabel("Frequency", fontsize = 10)
# ax2.set_ylabel("")

# plt.tight_layout(pad=2)
# plt.show()


# We do have the field discourse_type_num. We see that Evidence1, Position1 and Claim1 are almost always there in an essay. Most students also had at least one Concluding Statement. What's surprising to me is that a Lead is missing in about 40% of the essays (Lead 1 is found in almost 60% of the essays).
# 
# The graph only plots discourse_type_nums which are found in at least 3% of the essays.

# In[10]:


# STEFANOS: Disable plotting
# fig = plt.figure(figsize=(12,8))

# STEFANOS-DISABLE-FOR-MODIN:
#### ORIGINAL ####
# av_per_essay = train['discourse_type_num'].value_counts(ascending = True).rename_axis('discourse_type_num').reset_index(name='count')
#### CAN RUN WITH MODIN ####
av_per_essay = train['discourse_type_num'].value_counts(ascending = True)
av_per_essay.index.name = "discourse_type_num"
av_per_essay = av_per_essay.reset_index(name='count')

av_per_essay['perc'] = round((av_per_essay['count'] / train.id.nunique()),3)
av_per_essay = av_per_essay.set_index('discourse_type_num')
# ax = av_per_essay.query('perc > 0.03')['perc'].plot(kind="barh")
ax = av_per_essay.query('perc > 0.03')['perc']
# ax.set_title("discourse_type_num: Percent present in essays", fontsize=20, fontweight = 'bold')
# ax.bar_label(ax.containers[0], label_type="edge")
# ax.set_xlabel("Percent")
# ax.set_ylabel("")
# plt.show()


# Below you can see a plot with the average positions of the discourse start and end.

# In[11]:


data = train.groupby("discourse_type")[['discourse_end', 'discourse_start']].mean().reset_index().sort_values(by = 'discourse_start', ascending = False)
# data.plot(x='discourse_type',
#         kind='barh',
#         stacked=False,
#         title='Average start and end position absolute',
#         figsize=(12,4))
# plt.show()


# I am also interested in the relative positions of discourse types with the essays. Below you can see the distributions of the discourse types of the first and last discourses identified.

# In[12]:


# STEFANOS-DISABLE-FOR-MODIN:
#### ORIGINAL ####
#train_first = train.drop_duplicates(subset = "id", keep = "first").discourse_type.value_counts().rename_axis('discourse_type').reset_index(name='counts_first')
#### CAN RUN WITH MODIN ####
train_first = train.drop_duplicates(subset = "id", keep = "first").discourse_type.value_counts()
train_first.index.name = 'discourse_type'
train_first = train_first.reset_index(name='counts_first')

train_first['percent_first'] = round((train_first['counts_first']/train.id.nunique()),2)
# STEFANOS-DISABLE-FOR-MODIN:
#### ORIGINAL ####
# train_last = train.drop_duplicates(subset = "id", keep = "last").discourse_type.value_counts().rename_axis('discourse_type').reset_index(name='counts_last')
#### CAN RUN WITH MODIN ####
train_last = train.drop_duplicates(subset = "id", keep = "last").discourse_type.value_counts()
train_last.index.name = 'discourse_type'
train_last = train_last.reset_index(name='counts_last')


train_last['percent_last'] = round((train_last['counts_last']/train.id.nunique()),2)
train_first_last = train_first.merge(train_last, on = "discourse_type", how = "left")
train_first_last


# We also know that a Lead is missing in around 40% of the essays. Below you can see that if there is a Lead, it's almost always the first discourse identified in an essay (Lead 2 is very rare anyway).

# In[13]:


train['discourse_nr'] = 1
counter = 1

for i in (range(1, len(train))):
    if train.loc[i, 'id'] == train.loc[i-1, 'id']:
        counter += 1
        train.loc[i, 'discourse_nr'] = counter
    else:
        counter = 1
        train.loc[i, 'discourse_nr'] = counter

#if you are interested in other discourse_types you can add them to the list in df.query
# STEFANOS-DISABLE-FOR-MODIN:
# It seems you cannot call things on top of a groupby.
### ORIGINAL
# train.query('discourse_type in ["Lead"]').groupby('discourse_type_num')['discourse_nr'].value_counts().to_frame('occurences')
### COMPATIBLE WITH MODIN:
train.query('discourse_type in ["Lead"]').groupby('discourse_type_num')['discourse_nr']


# # Investigation the gaps between Annotations (text not used as discourse_text)

# Just taking the last discourse_end in train is not entirely correct as a last piece of text may not have been used as a discourse. Therefore, I will go through the essays to find the real ends. Eh....until I remembered that Rob Mulla already did that in the excellent EDA [Student Writing Competition [Twitch Stream]](https://www.kaggle.com/robikscube/student-writing-competition-twitch) ;-). Please upvote his notebook!

# In[14]:


# this code chunk is copied from Rob Mulla
len_dict = {}
word_dict = {}
for t in tqdm(train_txt):
    with open(t, "r") as txt_file:
        myid = t.split("/")[-1].replace(".txt", "")
        data = txt_file.read()
        mylen = len(data.strip())
        myword = len(data.split())
        len_dict[myid] = mylen
        word_dict[myid] = myword
train["essay_len"] = train["id"].map(len_dict)
train["essay_words"] = train["id"].map(word_dict)


# When comparing the discourse_end of the last discourse in each essay, we see that the discourse_end is sometimes larger than the essay_len. This cannot be right, but I will assume that those are last pieces of text in the essay indeed.

# In[15]:


#initialize column
train['gap_length'] = np.nan

#set the first one
train.loc[0, 'gap_length'] = 7 #discourse start - 1 (previous end is always -1)

#loop over rest
for i in tqdm(range(1, len(train))):
    #gap if difference is not 1 within an essay
    if ((train.loc[i, "id"] == train.loc[i-1, "id"])\
        and (train.loc[i, "discourse_start"] - train.loc[i-1, "discourse_end"] > 1)):
        train.loc[i, 'gap_length'] = train.loc[i, "discourse_start"] - train.loc[i-1, "discourse_end"] - 2
        #minus 2 as the previous end is always -1 and the previous start always +1
    #gap if the first discourse of an new essay does not start at 0
    elif ((train.loc[i, "id"] != train.loc[i-1, "id"])\
        and (train.loc[i, "discourse_start"] != 0)):
        train.loc[i, 'gap_length'] = train.loc[i, "discourse_start"] -1


 #is there any text after the last discourse of an essay?
last_ones = train.drop_duplicates(subset="id", keep='last')
last_ones['gap_end_length'] = np.where((last_ones.discourse_end < last_ones.essay_len),\
                                       (last_ones.essay_len - last_ones.discourse_end),\
                                       np.nan)

cols_to_merge = ['id', 'discourse_id', 'gap_end_length']
train = train.merge(last_ones[cols_to_merge], on = ["id", "discourse_id"], how = "left")


# In[16]:


#display an example
cols_to_display = ['id', 'discourse_start', 'discourse_end', 'discourse_type', 'essay_len', 'gap_length', 'gap_end_length']
train[cols_to_display].query('id == "AFEC37C2D43F"')


# In[17]:


#how many pieces of tekst are not used as discourses?
print(f"Besides the {len(train)} discourse texts, there are {len(train.query('gap_length.notna()', engine='python'))+ len(train.query('gap_end_length.notna()', engine='python'))} pieces of text not classified.")


# Although the gaps in the example above are small, we do have huge gaps in a number of essays.

# In[18]:


# STEFANOS: We have to change code slightly because we use less data. The original code plugs constant values
# which depend on some data existing.
if "IREWR_LESS_REPLICATION" in os.environ and os.environ["IREWR_LESS_REPLICATION"] == "True":
    _IREWR_tmp = train.sort_values(by = "gap_length", ascending = False)[cols_to_display].head()
    _IREWR_plug_2 = _IREWR_tmp.iloc[0]["id"]
    _IREWR_tmp
else:
    # Original
    train.sort_values(by = "gap_length", ascending = False)[cols_to_display].head()


# In[19]:


# STEFANOS: We have to change code slightly because we use less data. The original code plugs constant values
# which depend on some data existing.
if "IREWR_LESS_REPLICATION" in os.environ and os.environ["IREWR_LESS_REPLICATION"] == "True":
    _IREWR_tmp2 = train.sort_values(by = "gap_end_length", ascending = False)[cols_to_display].head()
    _IREWR_plug_1 = _IREWR_tmp2.iloc[1]["id"]
    _IREWR_tmp2
else:
    train.sort_values(by = "gap_length", ascending = False)[cols_to_display].head()


# Below, you can see a histogram of the length of all gaps with the outliers taken out (all gaps longer than 300 characters).

# In[21]:


all_gaps = (train.gap_length[~train.gap_length.isna()])._append((train.gap_end_length[~train.gap_end_length.isna()]), ignore_index= True)
#filter outliers
all_gaps = all_gaps[all_gaps<300]
# fig = plt.figure(figsize=(12,6))
# all_gaps.plot.hist(bins=100)
# plt.title("Histogram of gap length (gaps up to 300 characters only)")
# plt.xticks(rotation=0)
# plt.xlabel("Length of gaps in characters")
# plt.show()


# # Are there many really bad essays (large percentage of text not classified)?
# Yes, we do have those. Some have around 90% of text not classified as one of the discourse types.
# 
# Regarding the one with gap_end_length 7348: I found out that this student just copied and pasted the same texts multiple times in his/her essay. See discussion topic: [Finding: essay with all text repeated many times](https://www.kaggle.com/c/feedback-prize-2021/discussion/298193).

# In[22]:


total_gaps = train.groupby('id').agg({'essay_len': 'first',\
                                               'gap_length': 'sum',\
                                               'gap_end_length': 'sum'})
total_gaps['perc_not_classified'] = round(((total_gaps.gap_length + total_gaps.gap_end_length)/total_gaps.essay_len),2)

total_gaps.sort_values(by = 'perc_not_classified', ascending = False).head()


# # Color printing essays including the gaps
# 
# I saw  a very pretty way to do this in the Notebook made by Sanskar Hasija (https://www.kaggle.com/odins0n/feedback-prize-eda). The code is nice but did not print the gaps yet. Below, I make a function that adds all gaps in an essay as rows with discourse type "Nothing".

# In[23]:


def add_gap_rows(essay):
    cols_to_keep = ['discourse_start', 'discourse_end', 'discourse_type', 'gap_length', 'gap_end_length']
    df_essay = train.query('id == @essay')[cols_to_keep].reset_index(drop = True)
    
    print(df_essay)

    #index new row
    insert_row = len(df_essay)
   
    for i in range(1, len(df_essay)):          
        if df_essay.loc[i,"gap_length"] >0:
            if i == 0:
                start = 0 #as there is no i-1 for first row
                end = df_essay.loc[0, 'discourse_start'] -1
                disc_type = "Nothing"
                gap_end = np.nan
                gap = np.nan
                df_essay.loc[insert_row] = [start, end, disc_type, gap, gap_end]
                insert_row += 1
            else:
                start = df_essay.loc[i-1, "discourse_end"] + 1
                end = df_essay.loc[i, 'discourse_start'] -1
                disc_type = "Nothing"
                gap_end = np.nan
                gap = np.nan
                df_essay.loc[insert_row] = [start, end, disc_type, gap, gap_end]
                insert_row += 1

    df_essay = df_essay.sort_values(by = "discourse_start").reset_index(drop=True)

    #add gap at end
    if df_essay.loc[(len(df_essay)-1),'gap_end_length'] > 0:
        start = df_essay.loc[(len(df_essay)-1), "discourse_end"] + 1
        end = start + df_essay.loc[(len(df_essay)-1), 'gap_end_length']
        disc_type = "Nothing"
        gap_end = np.nan
        gap = np.nan
        df_essay.loc[insert_row] = [start, end, disc_type, gap, gap_end]
        
    return(df_essay)


# In[24]:


# STEFANOS: See above.
if "IREWR_LESS_REPLICATION" in os.environ and os.environ["IREWR_LESS_REPLICATION"] == "True":
    add_gap_rows(_IREWR_plug_1)
else:
    add_gap_rows("129497C3E0FC")


# This enables me to make a function that uses the code made by Sanskar Hasija to color print an essay including the gaps.

# In[25]:


def print_colored_essay(essay):
    df_essay = add_gap_rows(essay)
    #code from https://www.kaggle.com/odins0n/feedback-prize-eda, but adjusted to df_essay
    essay_file = "input/feedback-prize-2021/train/" + essay + ".txt"

    ents = []
    for i, row in df_essay.iterrows():
        ents.append({
                        'start': int(row['discourse_start']), 
                         'end': int(row['discourse_end']), 
                         'label': row['discourse_type']
                    })

    with open(essay_file, 'r') as file: data = file.read()

    doc2 = {
        "text": data,
        "ents": ents,
    }

    colors = {'Lead': '#EE11D0','Position': '#AB4DE1','Claim': '#1EDE71','Evidence': '#33FAFA','Counterclaim': '#4253C1','Concluding Statement': 'yellow','Rebuttal': 'red'}
    options = {"ents": df_essay.discourse_type.unique().tolist(), "colors": colors}
    # STEFANOS: Disable plotting-like code.
#     spacy.displacy.render(doc2, style="ent", options=options, manual=True, jupyter=True);


# In[26]:


# STEFANOS: See above.
# print_colored_essay("7330313ED3F0")
if "IREWR_LESS_REPLICATION" in os.environ and os.environ["IREWR_LESS_REPLICATION"] == "True":
    print_colored_essay(_IREWR_plug_2)
else:
    print_colored_essay("7330313ED3F0")


# # Most used words per Discourse Type
# 
# Initially, I did a manual effort to find out which single words were used most often.I took out stopwords, converted all text to lowercase, but left in the punctuation. I also took out some extra words that were all over the place in the figures for each discourse_type. After this effort, I was not sure how useful this is. One thing to notice is that "however," is used a lot in Rebuttal.
# 
# Later on, I decided that making one function for all n_grams was the way to go. If you are still interested in my manual effort for the single words, you can unhide the code in the cell below.

# In[31]:


train['discourse_text'] = train['discourse_text'].str.lower()

#get stopwords from nltk library
stop_english = stopwords.words("english")
other_words_to_take_out = ['school', 'students', 'people', 'would', 'could', 'many']
stop_english.extend(other_words_to_take_out)

#put dataframe of Top-10 words in dict for all discourse types
counts_dict = {}
for dt in train['discourse_type'].unique():
    df = train.query('discourse_type == @dt')
    text = df.discourse_text.apply(lambda x: x.split()).tolist()
    text = [item for elem in text for item in elem]
    df1 = pd.Series(text).value_counts().to_frame().reset_index()
    df1.columns = ['Word', 'Frequency']
    df1 = df1[~df1.Word.isin(stop_english)].head(10)
    df1 = df1.set_index("Word").sort_values(by = "Frequency", ascending = True)
    counts_dict[dt] = df1

# STEFANOS: Disable plotting
# plt.figure(figsize=(15, 12))
# plt.subplots_adjust(hspace=0.5)

keys = list(counts_dict.keys())

# STEFANOS: Disable plotting
# for n, key in enumerate(keys):
#     ax = plt.subplot(4, 2, n + 1)
#     ax.set_title(f"Most used words in {key}")
#     counts_dict[keys[n]].plot(ax=ax, kind = 'barh')
#     plt.ylabel("")

# plt.show()


# # Making n_grams for each discourse type
# 
# After the manual effort above, I was not fully pleased with the result and decided that I wanted to make a function to compose Top-10 n_grams per discount type by using CountVectorizer(). This function should also work for the single words (just run it with n_grams =1).

# In[35]:


def get_n_grams(n_grams, top_n = 10):
    df_words = pd.DataFrame()
    for dt in tqdm(train['discourse_type'].unique()):
        df = train.query('discourse_type == @dt')
        texts = df['discourse_text'].tolist()
        vec = CountVectorizer(lowercase = True, stop_words = 'english',\
                              ngram_range=(n_grams, n_grams)).fit(texts)
        bag_of_words = vec.transform(texts)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        cvec_df = pd.DataFrame.from_records(words_freq,\
                                            columns= ['words', 'counts']).sort_values(by="counts", ascending=False)
        cvec_df.insert(0, "Discourse_type", dt)
        cvec_df = cvec_df.iloc[:top_n,:]
        df_words = df_words._append(cvec_df)
    return df_words


# This function return one dataframe with 70 rows (the top 10 most used n-grams for each discourse type).

# In[36]:


bigrams = get_n_grams(n_grams = 2, top_n=10)
bigrams.head()


# Below, I have also made a function that prints the results in this dataframe as subplots.

# In[37]:


def plot_ngram(df, type = "bigrams"):
# STEFANOS: Disable plotting
#     plt.figure(figsize=(15, 12))
#     plt.subplots_adjust(hspace=0.5)

#     for n, dt in enumerate(df.Discourse_type.unique()):
#         ax = plt.subplot(4, 2, n + 1)
#         ax.set_title(f"Most used {type} in {dt}")
        data = df.query('Discourse_type == @dt')[['words', 'counts']].set_index("words").sort_values(by = "counts", ascending = True)
#         data.plot(ax=ax, kind = 'barh')
#         plt.ylabel("")
#     plt.tight_layout()
#     plt.show()
    
plot_ngram(bigrams)


# Below, I am also plotting the trigrams using both functions in one go.

# In[38]:


trigrams = get_n_grams(n_grams = 3, top_n=10)
plot_ngram(trigrams, type = "trigrams")


# # NER Introduction
# 
# Named Entity Recognition (NER) is the technique that works best for this challenge. If you are looking for more info on this, the free course on Hugging Face is strongly recommended. In the section [Token Classification](https://huggingface.co/course/chapter7/2?fw=pt), we can find the following things that are relevant here:
# - Named entity recognition (NER): Find the entities (such as persons, locations, or organizations) in a sentence. This can be formulated as attributing a label to each token by having one class per entity and one class for “no entity.”
# - Chunking: Find the tokens that belong to the same entity. This task (which can be combined with POS or NER) can be formulated as attributing one label (usually B-) to any tokens that are at the beginning of a chunk, another label (usually I-) to tokens that are inside a chunk, and a third label (usually O) to tokens that don’t belong to any chunk.
# 
# Basically, what is being used in this competition is NER Chunking. Darek Kłeczek wrote a great notebook that explains the ideas behind this (please upvote!): [Visual Tutorial NER Chunking Token Classification](https://www.kaggle.com/thedrcat/visual-tutorial-ner-chunking-token-classification).
# 
# In this section, I am only going to show how these NER labels can be made for this competition. I am basically using the loop found in Chris Deotte's great notebook [PyTorch - BigBird - NER - [CV 0.615]](https://www.kaggle.com/cdeotte/pytorch-bigbird-ner-cv-0-615) (Please upvote his notebook!), but tried to make it a little easier to understand. I am also using df.loc instead of df.iterrows.
# 
# First, we have to make a dataframe with all full texts of the essays in a dataframe.

# In[39]:


# https://www.kaggle.com/raghavendrakotala/fine-tunned-on-roberta-base-as-ner-problem-0-533
test_names, train_texts = [], []
for f in tqdm(list(os.listdir('input/feedback-prize-2021/train'))):
    test_names.append(f.replace('.txt', ''))
    train_texts.append(open('input/feedback-prize-2021/train/' + f, 'r').read())
train_text_df = pd.DataFrame({'id': test_names, 'text': train_texts})
train_text_df.head()


# Now we are ready to add a column with NER entities to this dataframe.

# In[40]:


all_entities = []
#loop over dataframe with all full texts
for i in tqdm(range(len(train_text_df))):
    total = len(train_text_df.loc[i, 'text'].split())
    #now a list with length the total number of words in an essay is initialised with all values being "O"
    entities = ["O"]*total
    #now loop over dataframe with all discourses of this particular essay
    discourse_id = train_text_df.loc[i, 'id']
    train_df_id = train.query('id == @discourse_id').reset_index(drop=True)
    for j in range(len(train_df_id)):
        discourse = train_df_id.loc[j, 'discourse_type']
        #make a list with the position numbers in predictionstring converted into integer
        list_ix = [int(x) for x in train_df_id.loc[j, 'predictionstring'].split(' ')]
        #now the entities lists gets overwritten where there are discourse identified by the experts
        #the first word of each discourse gets prefix "Beginning"
        entities[list_ix[0]] = f"B-{discourse}"
        #the other ones get prefix I
        for k in list_ix[1:]: entities[k] = f"I-{discourse}"
    all_entities.append(entities)
    
    
train_text_df['entities'] = all_entities


# In[41]:


train_text_df.head()


# **Thanks for your attention! If you like this notebook, an upvote is always appreciated.**
