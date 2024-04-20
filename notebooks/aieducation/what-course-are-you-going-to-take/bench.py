#!/usr/bin/env python
# coding: utf-8

# <center><h1> 🏫What course are your going to take next semester?📚 </h1></center>

# #### What is your favorite course or class? Or what was your favorite course or class? I loved computer science course and sports courses. When I was in university, it was really hard to choose which class to take next semester. Because I didn't want to waste of my time. What methods did you use to choose which courses or classes to take next semester? How about this method? Let's check it out!

# ![dom-fou-YRMWVcdyhmI-unsplash.jpg](attachment:ac184e1a-3302-484a-a2f1-b7b6667de5d1.jpg)

# <a id="intro"></a>
# # Data Introduction💁
# - **course_code:** Course code at the University of Waterloo as stated on uwflow.com
# - **course_title:** Title of the course
# - **num_ratings:** number of course reviews (a review need not contain text, only feedback such as "useful", "easy", etc..)
# - **useful:** Percent of reviewers that said the course was useful
# - **easy:** Percent of reviewers that said the course was easy
# - **liked:** Percent of reviewers that said they liked the course
# - **num_reviews:** Number of reviews for the course that included a text review
# - **reviews:** Course reviews
# - **course_rating:** Whether or not the reviewer said they liked the course
# - **course_rating_int:** Whether or not the reviewer said they liked the course, as an integer.

# # 🏫Table of Contents👩‍🎓
# - [1. Import Libraries](#import)
# - [2. Check Data and Preparation](#check)
# - [3. EDA & Visualization](#eda)
#     - [3-1. Ranking Graph](#ranking)
# - [4. KOREA Course](#korea): Top rated course
# - [5. CHINA Course](#china): Second rated course
# - [6. SPANISH Course](#spanish): Third rated course
# - [7. Computer Science Course](#computer)
# - [8. Work-term Report Course](#work): Worst rated course
# - [9. Professional Development Course](#pd): Second worst rated course

# <a id="import"></a>
# # 1️⃣ㅣImport Libraries

# In[106]:


import os
# STEFANOS: Conditionally import Modin Pandas
import pandas as pd
import numpy as np


# In[ ]:


def hash_anything(obj):
    import pandas.util
    if isinstance(obj, pd.DataFrame):
        return pandas.util.hash_pandas_object(obj, index=False).to_numpy().data
    elif isinstance(obj, np.ndarray):
        return obj.data
    elif isinstance(obj, list):
        return str(obj).encode()
    else:
        return str(obj).encode()


def hash_dataframe(df):
    import xxhash
    h = xxhash.xxh64()
    for column in df.round(6).columns:
        h.update(hash_anything(df[column]))
    return h.digest()


# <a id="check"></a>
# # 2️⃣ㅣCheck Data and Preparation💾

# In[107]:


course = pd.read_csv(os.path.abspath('') + "/input/course-reviews-university-of-waterloo/course_data_clean.csv")


# # -- STEFANOS -- Replicate Data

# In[108]:


factor = 300
course = pd.concat([course]*factor, ignore_index=True)
# course.info()


# In[109]:


course.shape


# In[110]:


course.head(10)


# In[111]:


course.tail(10)


# In[112]:


# course.describe()


# In[113]:


# course.info()


# ### We don't need data which has no course rating int value. So drop it.

# In[114]:


course = course.dropna()


# ### Extract course unit and course number from course code. I'm going to group by course code so I need it.

# In[115]:


course[["course_unit", "course_num"]] = course["course_code"].str.split(" ", expand=True)


# In[116]:


course


# ### If number of people who rated is under 10, I think that data is not reliable. I can't ruin my semester because of those unreliable data!

# In[117]:


course[course["num_reviews"] < 10].index


# In[118]:


course.drop(course[course["num_reviews"] < 10].index, inplace=True)


# In[119]:


course


# ### I'm going to look relationship between useful, easy, liked and course rating score. So eliminate %.

# In[120]:


for i in ["useful", "easy", "liked"]:
    course[i] = course[i].str.replace("%", "")
    course[i] = course[i].astype("int")


# In[121]:


course.set_index("course_unit", inplace=True)


# In[122]:


course


# ### We don't need course title because we have course code. And we don't need reviews and course_rating.

# In[123]:


course.drop(["course_title", "reviews", "course_rating"], axis=1, inplace=True)


# In[124]:


course


# In[125]:


# course.info()


# In[126]:


course_gp = course.groupby("course_unit").mean(numeric_only=True)


# In[127]:


course_gp


# In[128]:


# STEFANOS-DISABLE-FOR-MODIN: Modin seems to fail in the LHS, giving an error that "course_rating_mean" is
# not in the DF. Of course, it's valid code because we're creating this column here.
# The problem we cannot just disable this code because follow-up code depends on this column. So, we will create
# the column before the loop so that indexing in the LHS does not fail.

###### ORIGINAL CODE ###########
# for i in course_gp.index:
#     course.loc[i, "course_rating_mean"] = course_gp.loc[i, "course_rating_int"]

###### CODE THAT MODIN CAN RUN ########
course["course_rating_mean"] = None
for i in course_gp.index:
    course.loc[i, "course_rating_mean"] = course_gp.loc[i, "course_rating_int"]


# In[ ]:


print(hash_dataframe(course))


# In[129]:


course


# In[130]:


course.reset_index(inplace=True)


# In[131]:


course


# In[132]:


course.groupby("course_unit").mean(numeric_only=True)["course_rating_int"]


# In[133]:


course[course["course_code"].str.startswith("CS")].value_counts()


# In[134]:


course


# <a id="eda"></a>
# # 3️⃣ㅣEDA & Visualization📊

# ### Let's see relationshtp overview between variables.

# In[135]:


# STEFANOS: Disable plotting
# plt.figure(figsize=(25,20))
# sns.pairplot(data=course)
# plt.show()


# In[136]:


# STEFANOS: Disable plotting
# plt.figure(figsize=(12,10))
# sns.scatterplot(x="num_ratings", y="course_rating_mean", data=course)
# plt.show()


# In[137]:


# STEFANOS: Disable plotting
# plt.figure(figsize=(12,10))
# sns.scatterplot(x="num_reviews", y="course_rating_mean", data=course)
# plt.show()


# In[138]:


# STEFANOS: Disable plotting
# plt.figure(figsize=(25,20))
# sns.scatterplot(x="useful", y="course_rating_mean", hue="course_unit", data=course)
# plt.xlabel("useful", fontsize = 20)
# plt.ylabel("Course Rating Mean", fontsize = 20)
# plt.show()


# ### The two variables have a direct proportional relationship.

# In[139]:


# STEFANOS: Disable plotting
# plt.figure(figsize=(25,20))
# sns.scatterplot(x="easy", y="course_rating_mean", hue="course_unit", data=course)
# plt.xlabel("easy", fontsize = 20)
# plt.ylabel("Course Rating Mean", fontsize = 20)
# plt.show()


# ### The two variables have a not strong relatioship.

# In[140]:


# STEFANOS: Disable plotting
# plt.figure(figsize=(25,20))
# sns.scatterplot(x="liked", y="course_rating_mean", hue="course_unit", data=course)
# plt.xlabel("liked", fontsize = 20)
# plt.ylabel("Course Rating Mean", fontsize = 20)
# plt.show()


# ### The two variables have a direct proportional relationship

# <a id="ranking"></a>
# ## Ranking Graph 📊

# In[141]:


course.sort_values("course_rating_mean", ascending=False, inplace=True)

# STEFANOS: Disable plotting
# plt.figure(figsize=(25,20))
# sns.set_palette("Pastel1")
# sns.barplot(x="course_rating_mean", y="course_unit", data=course)
# plt.xlabel("Course Rating Mean", fontsize = 20)
# plt.ylabel("Course Unit", fontsize = 20)
# plt.show()


# ### We can see language courses are top rated. And top of top is KOREA course. Bottom of bottom is Work-term Report course. Let's check it out more.

# ### Let's check top rated courses, computer science course and bottom courses.

# <a id="korea"></a>
# # 4️⃣ㅣKOREA Course: Ranking 1st place

# In[142]:


course.reset_index(inplace=True)


# In[143]:


course.set_index("course_unit", inplace=True)


# In[144]:


course.loc["KOREA", "course_rating_mean"].value_counts()


# In[145]:


KOREA = course.loc["KOREA", :]


# In[146]:


# STEFANOS: Disable plotting
# plt.figure(figsize=(10,10))
# sns.barplot(x="course_code", y="course_rating_mean", data=KOREA)
# plt.xlabel("KOREA course", fontsize = 20)
# plt.ylabel("Course Rating Mean", fontsize = 20)
# plt.show()


# ### Korea course has just one class and all students answered it was great.

# <a id="china"></a>
# # 5️⃣ㅣCHINA Course: Ranking 2nd place

# In[147]:


course.loc["CHINA", "course_rating_mean"].value_counts()


# In[148]:


china = course.loc["CHINA", :]


# In[149]:


# STEFANOS: Disable plotting
# plt.figure(figsize=(10,10))
# sns.barplot(x="course_code", y="course_rating_mean", data=china)
# plt.xlabel("CHINA course", fontsize = 20)
# plt.ylabel("Course Rating Mean", fontsize = 20)
# plt.show()


# ### China course has only one class too, and students answered it was great.

# <a id="spanish"></a>
# # 6️⃣ㅣSPANISH Course: Ranking 3rd place

# In[150]:


course.loc["CHINA", "course_rating_mean"].value_counts()


# In[151]:


span = course.loc["SPAN", :]

# STEFANOS: Disable plotting
# plt.figure(figsize=(10,10))
# sns.barplot(x="course_code", y="course_rating_mean", data=span)
# plt.xlabel("SPANISH course", fontsize = 20)
# plt.ylabel("Course Rating Mean", fontsize = 20)
# plt.show()


# ### Spanish course has only one class too. But not all students answered it was great. Still it is top rated course. If you hesitate to take spanish course, don't!

# <a id="computer"></a>
# # 7️⃣ㅣComputer Science Course

# In[152]:


course.loc["CS", "course_rating_mean"].value_counts()


# In[153]:


cs = course.loc["CS", :]
cs


# In[154]:


cs_mean = cs.groupby("course_code").mean(numeric_only=True).sort_values("course_rating_int", ascending=False)
cs_mean


# In[155]:


# STEFANOS: Disable plotting
# plt.figure(figsize=(15,10))
# sns.barplot(x="course_rating_int", y=cs_mean.index, data=cs_mean)
# plt.xlabel("Course Rating Mean", fontsize = 20)
# plt.ylabel("CS Course", fontsize = 20)
# plt.show()


# ### Computer Science Course has many classes. So I lined up all classes by rating score mean. If you want to take Computer Science course, with this data you will be able to help decide.

# <a id="work"></a>
# # 8️⃣ㅣWork-term Report: Worst rated

# In[156]:


course.loc["WKRPT", "course_rating_mean"].value_counts()


# In[157]:


wkrpt = course.loc["WKRPT", :]
wkrpt


# In[158]:


wkrpt_mean = wkrpt.groupby("course_code").mean(numeric_only=True).sort_values("course_rating_int", ascending=False)
wkrpt_mean


# In[159]:


# STEFANOS: Disable plotting
# plt.figure(figsize=(15,10))
# sns.barplot(x="course_rating_int", y=wkrpt_mean.index, data=wkrpt_mean)
# plt.xlabel("Course Rating Mean", fontsize = 20)
# plt.ylabel("WKRPT Course", fontsize = 20)
# plt.show()


# ### Wow, there is a class that no one answered it was good. We can avoid that class. And another one also has very low rating score.

# <a id="pd"></a>
# # 9️⃣ㅣProfessional Development Course: Second Worst rated

# In[160]:


course.loc["PD", "course_rating_mean"].value_counts()


# In[161]:


pd = course.loc["PD", :]
pd


# In[162]:


pd_mean = pd.groupby("course_code").mean(numeric_only=True).sort_values("course_rating_int", ascending=False)
pd_mean


# In[163]:


# STEFANOS: Disable plotting
# plt.figure(figsize=(15,10))
# sns.barplot(x="course_rating_int", y=pd_mean.index, data=pd_mean)
# plt.xlabel("Course Rating Mean", fontsize = 20)
# plt.ylabel("PD Course", fontsize = 20)
# plt.show()


# ### We can see in one course, there is big difference. I think this is why we need to analyze data!

# <center><h1>If you want to check other courses, copy and edit this kernel! If it was fun and imformative please UPVOTE!👍 Thank you!</h1></center>
