#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


# STEFANOS: Disable pip install
# !pip install lazypredict
# !pip install --upgrade pandas
# !pip install fast-tabnet
# !pip install fastai
# !pip install pandas-profiling


# In[3]:


#A program that takes a csv and trains models on it. Streamlined model selection.
#==============================================================================

# STEFANOS: Disable unneeded modules
# #LazyPredict
# import lazypredict
# from lazypredict.Supervised import LazyRegressor
# from lazypredict.Supervised import LazyClassifier
# #Baysian Optimization
# from bayes_opt import BayesianOptimization
#Pandas stack
import os
# STEFANOS: Conditionally import Modin Pandas
import pandas as pd
import numpy as np
# STEFANOS: Disable unneeded modules
# #FastAI
from fastai.tabular.all import *
from fastai.tabular.core import *
# #Plots
# import matplotlib.pyplot as plt
# import seaborn as sns
#System
import os
import sys
import traceback
#Fit an xgboost model
# STEFANOS: Disable unneeded modules
# from xgboost import XGBRegressor
# from xgboost import XGBClassifier
# from xgboost import plot_importance
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import roc_auc_score
#Random
import random

#TabNet
from fast_tabnet.core import *

import shutil


# In[4]:


# STEFANOS: Disable matplotlib inline
# %matplotlib inline


# In[5]:


# For Styling
# STEFANOS: Disbale plotting
# plt.style.use('seaborn-bright')


# In[6]:


#Project Variables
#===================================================================================================
PROJECT_NAME = 'superstore'
VARIABLE_FILES = False
#Maximum amount of rows to take

## -- STEFANOS -- Remove sample

# SAMPLE_COUNT = 4000
FASTAI_LEARNING_RATE = 1e-1
AUTO_ADJUST_LEARNING_RATE = False
#Set to True automatically infer if variables are categorical or continuous
ENABLE_BREAKPOINT = True
#When trying to declare a column a continuous variable, if it fails, convert it to a categorical variable
CONVERT_TO_CAT = False
REGRESSOR = True
SEP_DOLLAR = False
SEP_COMMA = True
SHUFFLE_DATA = True


# In[7]:


input_dir = f'input/{PROJECT_NAME}'
# STEFANOS: Change the working path
# param_dir = f'/kaggle/working/{PROJECT_NAME}'
param_dir = f'./working/{PROJECT_NAME}'
TARGET = ''
PARAM_DIR = param_dir
print(f'param_dir: {param_dir}')
if not os.path.exists(param_dir):
    os.makedirs(param_dir)
#rename any file in param_dir/file that ends with csv to data.csv
for file in os.listdir(input_dir):
    if file.endswith('.csv'):
        print('CSV!')
        if 'classification_results' not in file and 'regression_results' not in file:
            #os.rename(f'{input_dir}/{file}', f'{param_dir}/data.csv')
            shutil.copy(f'{input_dir}/{file}', f'{param_dir}/data.csv')
        #os.rename(f'{param_dir}/{file}', f'{param_dir}/data.csv')
try:
    # STEFANOS: Remove sample
#     df = pd.read_csv(f'{param_dir}/data.csv', nrows=SAMPLE_COUNT)
    df = pd.read_csv(f'{param_dir}/data.csv')
except:
    print(f'Please place a file named data.csv in {param_dir}')
    #sys.exit()


# In[ ]:





# In[8]:


df


# # -- STEFANOS -- Replicate Data

# In[9]:


factor = 10
df = pd.concat([df]*factor)
# df.info()


# In[21]:


# if SEP_DOLLAR:
#     #For every column in df, if the column contains a $, make a new column with the value without the $
#     for col in df.columns:
#         if '$' in df[col].to_string():
#             df[col + '_no_dollar'] = df[col].str.replace('$', '').str.replace(',', '')
#             #Try to convert this new column to a numeric type
#             try:
#                 df[col + '_no_dollar'] = df[col + '_no_dollar'].apply(pd.to_numeric, errors='coerce').dropna()
#             except Exception:
#                 print(f'{col} can not be converted to a float!')


# if SEP_COMMA:
#For every column in df, if the column contains a %, make a new column with the value without the %
for col in df.columns:
    if '%' in df[col].to_string() or ',' in df[col].to_string():
        df[col + '_processed'] = df[col].str.replace('%', '').str.replace(',', '')
        #Try to convert this new column to a numeric type
        try:
            df[col + '_processed'] = df[col + '_processed'].apply(pd.to_numeric, errors='coerce').dropna()
        except Exception:
            print(f'{col} can not be converted to a float!')


# In[ ]:





# In[11]:


df


# In[12]:


df.isna().sum()


# In[13]:


# df.profile_report()


# In[15]:


# STEFANOS: Disable plotting
# sns.heatmap(df.corr())
# _ = df.corr()


# In[17]:


# STEFANOS: Disable plotting
# df.head().style.background_gradient(cmap = "inferno")
df.head()


# In[18]:


# STEFANOS: Disable plotting
# df.describe().T.style.background_gradient(cmap = "viridis")
# df.describe().T


# In[19]:


df.columns


# In[22]:


target = ''
target_str = ''
#The column closest to the end isPARAM_DIR the target variable that can be represented as a float is the target variable
targets = []
#Loop through every possible target column (Continuous)
for i in range(len(df.columns)-1, 0, -1):
    try:
        df[df.columns[i]] = df[df.columns[i]].apply(pd.to_numeric, errors='coerce').dropna()
        target = df.columns[i]
        target_str = target.replace('/', '-')
    except:
        continue
    print(f'Target Variable: {target}')
    #Will be determined by the file name


    #===================================================================================================

    #Create project config files if they don't exist.
    if not os.path.exists(param_dir):
        #create param_dir
        os.makedirs(PARAM_DIR)
    if not os.path.exists(f'{PARAM_DIR}/cats.txt'):
        #create param_dir
        with open(f'{PARAM_DIR}/cats.txt', 'w') as f:
            f.write('')
    if not os.path.exists(f'{PARAM_DIR}/conts.txt'):
        #create param_dir
        with open(f'{PARAM_DIR}/conts.txt', 'w') as f:
            f.write('')
    if not os.path.exists(f'{PARAM_DIR}/cols_to_delete.txt'):
        with open(f'{PARAM_DIR}/cols_to_delete.txt', 'w') as f:
            f.write('')

    df = df.drop_duplicates()
    if SHUFFLE_DATA:
        df = df.sample(frac=1).reset_index(drop=True)

    # workaround for fastai/pytorch bug where bool is treated as object and thus erroring out.
    for n in df:
        if pd.api.types.is_bool_dtype(df[n]):
            df[n] = df[n].astype('uint8')

    with open(f'{PARAM_DIR}/cols_to_delete.txt', 'r') as f:
        cols_to_delete = f.read().splitlines()
    for col in cols_to_delete:
        try:
            del(df[col])
        except:
            pass
    #try to fill in missing values now, otherwise FastAI will do it for us later
    try:
        df = df.fillna(0)
    except:
        pass
    #print missing values
    #print(df.isna().sum().sort_values(ascending=False))
    #shrink df as much as possible
    df = df_shrink(df)


    #print types inside of df
    #print(df.dtypes)


    #Auto detect categorical and continuous variables
    #==============================================================================
    likely_cat = {}
    for var in df.columns:
        likely_cat[var] = 1.*df[var].nunique()/df[var].count() < 0.05 #or some other threshold

    cats = [var for var in df.columns if likely_cat[var]]
    conts = [var for var in df.columns if not likely_cat[var]]

    #remove target from lists
    try:
        conts.remove(target)
        cats.remove(target)
    except:
        pass
    #Convert target to float
    df[target] = df[target].apply(pd.to_numeric, errors='coerce').dropna()

    print('CATS=====================')
    print(cats)
    print('CONTS=====================')
    print(conts)

    #Populate categorical and continuous lists
    #==============================================================================

    if VARIABLE_FILES == True:
        with open(f'{PARAM_DIR}/cats.txt', 'r') as f:
            cats = f.read().splitlines()

        with open(f'{PARAM_DIR}/conts.txt', 'r') as f:
            conts = f.read().splitlines()

    #==============================================================================

    #==============================================================================
    procs = [Categorify, FillMissing, Normalize]
    #print(df.describe().T)
    # STEFANOS: Remove samples
#     df = df[0:SAMPLE_COUNT]
    splits = RandomSplitter()(range_of(df))

    print((len(cats)) + len(conts))
    #conts = []

    #Convert cont variables to floats
    #==============================================================================

    #Convert cont variables to floats
    #==============================================================================

    for var in conts:
        try:
            df[var] = df[var].apply(pd.to_numeric, errors='coerce').dropna()
        except:
            print(f'Could not convert {var} to float.')
            pass

    #==============================================================================

    #Experimental logic to add columns one-by-one to find a breakpoint
    #==============================================================================
    # if ENABLE_BREAKPOINT == True:
    temp_procs = [Categorify, FillMissing]
    print('Looping through continuous variables to find breakpoint')
    cont_list = []
    for cont in conts:
        focus_cont = cont
        cont_list.append(cont)
        #print(focus_cont)
        try:
            to = TabularPandas(df, procs=procs, cat_names=cats, cont_names=cont_list, y_names=target, y_block=RegressionBlock(), splits=splits)
            del(to)
        except:
            print('Error with ', focus_cont)
            #remove focus_cont from list
            cont_list.remove(focus_cont)
            #traceback.print_exc()
            continue
    #convert all continuous variables to floats
    for var in cont_list:
        try:
            df[var] = df[var].apply(pd.to_numeric, errors='coerce').dropna()
        except:
            print(f'Could not convert {var} to float.')
            cont_list.remove(var)
            if CONVERT_TO_CAT == True:
                cats.append(var)
            pass
    print(f'Continuous variables that made the cut : {cont_list}')
    print(f'Categorical variables that made the cut : {cats}')
    #shrink df as much as possible
    df = df_shrink(df)
    #print(df.dtypes)

    #==============================================================================

    #Creating tabular object + quick preprocessing
    #==============================================================================
    to = None
    # if REGRESSOR == True:
    try:
        to = TabularPandas(df, procs, cats, conts, target, y_block=RegressionBlock(), splits=splits)
    except:
        conts = []
        to = TabularPandas(df, procs, cats, conts, target, y_block=RegressionBlock(), splits=splits)
    # else:
    #     try:
    #         to = TabularPandas(df, procs, cats, conts, target, splits=splits)
    #     except:
    #         conts = []
    #         to = TabularPandas(df, procs, cats, conts, target, splits=splits)

### STEFANOS - Disabled ML code ###
            
#     #print(dir(to))
#     #print(to.xs)
#     dls = to.dataloaders()
#     print(f'Tabular Object size: {len(to)}')
#     try:
#         dls.one_batch()
#     except:
#         print(f'problem with getting one batch of {PROJECT_NAME}')
#     #==============================================================================

#     #Extracting train and test sets from tabular object
#     #==============================================================================

#     X_train, y_train = to.train.xs, to.train.ys.values.ravel()
#     X_test, y_test = to.valid.xs, to.valid.ys.values.ravel()

#     #Make sure target isn't in independent columns
#     if target in X_train and target in X_test:
#         del(X_train[target])
#         del(X_test[target])
#     #create dataframe from X_train and y_train
#     #export tabular object to csv
#     pd.DataFrame(X_train).to_csv(f'{PARAM_DIR}/X_train_{target_str}.csv', index=False)
#     pd.DataFrame(X_test).to_csv(f'{PARAM_DIR}/X_test_{target_str}.csv', index=False)
#     pd.DataFrame(y_train).to_csv(f'{PARAM_DIR}/y_train_{target_str}.csv', index=False)
#     pd.DataFrame(y_test).to_csv(f'{PARAM_DIR}/y_test_{target_str}.csv', index=False)

#     #==============================================================================

#     #==============================================================================

#     #Ready for model selection!

#     if REGRESSOR == True:
#         try:
#             reg = LazyRegressor(verbose=2, ignore_warnings=False, custom_metric=None)
#             models, predictions = reg.fit(X_train, X_test, y_train, y_test)
#             print(f'Project: {PROJECT_NAME}')
#             print(PROJECT_NAME)
#             print(f'Target: {target}')
#             print(target)
#             target_std = y_train.std()
#             print(f'Target Standard Deviation: {target_std}')
#             print(models)
#             models['project'] = PROJECT_NAME
#             models['target'] = target
#             models['target_std'] = target_std
#             #rename index of 
#             models.to_csv(f'{PARAM_DIR}/regression_results_{target_str}.csv', mode='a', header=True, index=True)
#         except:
#             print('Issue during lazypredict analysis')
#     else:
#         #TODO: remove this
#         try:
#             clf = LazyClassifier(verbose=2, ignore_warnings=False, custom_metric=None)
#             models, predictions = clf.fit(X_train, X_test, y_train, y_test)
#             print(f'Project: {PROJECT_NAME}')
#             print(PROJECT_NAME)
#             print(f'Target: {target}')
#             print(target)
#             print(f'Target Standard Deviation: {y_train.std()}')
#             print(models)
#             models.to_csv(f'{PARAM_DIR}/classification_results.csv', mode='a', header=False)
#         except:
#             print('Issue during lazypredict analysis')

#     model_name = 'tabnet'

#     # FastAI + pre-trained TabNet
#     #==============================================================================
#     learn = None
#     i = 0
#     while True:
#         try:
#             del learn
#         except:
#             pass
#         try:
#             learn = 0
#             model = TabNetModel(get_emb_sz(to), len(to.cont_names), dls.c, n_d=64, n_a=64, n_steps=5, virtual_batch_size=256)
#             # save the best model so far, determined by early stopping
#             cbs = [SaveModelCallback(monitor='_rmse', comp=np.less, fname=f'{model_name}_{PROJECT_NAME}_{target_str}_best'), EarlyStoppingCallback()]
#             learn = Learner(dls, model, loss_func=MSELossFlat(), metrics=rmse, cbs=cbs)
#             #learn = get_learner(to)
#             if(learn != 0):
#                 break
#             if i > 50:
#                 break
#         except:
#             i += 1
#             print('Error in FastAI TabNet')
#             traceback.print_exc()
#             continue
#     try:
#         #display learning rate finder results
#         x = learn.lr_find()
#     except:
#         pass
#     if AUTO_ADJUST_LEARNING_RATE == True:
#         FASTAI_LEARNING_RATE = x.valley
#     print(f'LEARNING RATE: {FASTAI_LEARNING_RATE}')
#     try:
#         if i < 50:
#             learn.fit_one_cycle(20, FASTAI_LEARNING_RATE)
#             plt.figure(figsize=(10, 10))
#             try:
#                 ax = learn.show_results()
#                 plt.show(block=True)
#             except:
#                 print('Could not show results')
#                 pass
#     except:
#         print('Could not fit model')
#         traceback.print_exc()
#         pass

#     #==============================================================================

#     #fit an xgboost model
#     #==============================================================================
#     if REGRESSOR == True:
#         xgb = XGBRegressor()
#     else:
#         xgb = XGBClassifier()
#     try:
#         xgb = XGBRegressor()
#         xgb.fit(X_train, y_train)
#         y_pred = xgb.predict(X_test)
#         print('XGBoost Predictions vs Actual==========')
#         print(pd.DataFrame({'actual': y_test, 'predicted': y_pred}).head())
#         print('XGBoost RMSE: ', np.sqrt(mean_squared_error(y_test, y_pred)))
#         #save feature importance plot to file
#         plot_importance(xgb)
#         plt.title(f'XGBoost Feature Importance for {PROJECT_NAME} | Target : {target}', wrap=True)
#         plt.tight_layout()
#         plt.show()
#         plt.savefig(f'{PARAM_DIR}/xgb_feature_importance_{target_str}.png')
#         fi_df = pd.DataFrame([xgb.get_booster().get_score()]).T
#         fi_df.columns = ['importance']
#         #create a column based off the index called feature
#         fi_df['feature'] = fi_df.index
#         #create a dataframe of feature importance
#         fi_df = fi_df[['feature', 'importance']]
#         fi_df.to_csv(f'{PARAM_DIR}/xgb_feature_importance_{target_str}.csv', index=False)
#         #xgb_fi = pd.DataFrame(xgb.feature_importances_, index=X_train.columns, columns=['importance'])
#         #xgb_fi.to_csv(f'{PARAM_DIR}/xgb_feature_importance_{target_str}.csv')
#         #print('XGBoost AUC: ', roc_auc_score(y_test, y_pred))
#     except:
#         traceback.print_exc()
#         print('XGBoost failed')


# # -- STEFANOS -- We have to disable the following too because even though it's Pandas code, it uses results from ML

# In[ ]:


# out_dir = f'./{PROJECT_NAME}'
# xgb_feature_importance_csvs = []

# for file in os.listdir(out_dir):
#     if 'xgb_feature_importance' in file and '.csv' in file:
#         xgb_feature_importance_csvs.append(pd.read_csv(os.path.join(out_dir, file)))

# xgb_feature_importance = pd.concat(xgb_feature_importance_csvs,axis=0)
# xgb_feature_importance.rename(columns={'Unnamed: 0': 'feature'}, inplace=True)
# print(xgb_feature_importance.head())
# xgb_feature_importance.groupby('feature')['importance'].mean().sort_values(ascending=False).plot(kind='bar', title='XGBoost Overall Feature Importance', figsize=(20, 10))


# In[ ]:


df.isna().sum()


# # **To Be Continued...**
