#!/usr/bin/env python
# coding: utf-8

# # Creating Player Stats Using Tracking Data: Snap Speed and Penetration 
# An important use of the player tracking data is "feature engineering", or creating new features using the data available. In this notebook, I will describe a simple set of metrics using the player tracking information in order to create some "season averages" of players. These kinds of player statistics can help explain the underlying talent of a player. 

# # Load packages

# In[1]:


import os
# STEFANOS: Conditionally import Modin Pandas
import pandas as pd


# # Load one week of player tracking data and PFF scouting data

# In[2]:


data = pd.read_csv(os.path.abspath('') + '/input/nfl-big-data-bowl-2023/week1.csv')
scout = pd.read_csv(os.path.abspath('') + '/input/nfl-big-data-bowl-2023/pffScoutingData.csv')
plays = pd.read_csv(os.path.abspath('') + '/input/nfl-big-data-bowl-2023/plays.csv')
players = pd.read_csv(os.path.abspath('') + '/input/nfl-big-data-bowl-2023/players.csv')

# Let's merge these data into one set 
data = data.merge(scout, how='left')
data.shape


# # -- STEFANOS -- Replicate Data
# 
# In fact, in this one, `data` seems to be big enough that I think it's ok to not replicate it.

# In[3]:


# data.info()


# # Create snap metrics 
# Let's use the tracking data to create metrics based around the snap. After 500ms of the snap, how fast is someone going? How far from the line of scrimmage is a player? This can help us gain more insights on how quickly a player gets off the line of scrimmage and also their depth relative to the line of scrimmage. 

# In[4]:


# get ball snap indicies 
_idxs = (data
         .loc[data['event']=='ball_snap', 
              'frameId']
         .index
         .values)

# to get 500ms of movement after snap, get 5 rows (each row is 100ms of info)
x = [(_idxs+x).tolist() for x in range(0,6)]
idxs = [item for sublist in x for item in sublist] #the output x is a list of lists, so this is just to flatten the list

# filter for snap + 500ms of data only using our selected indicies
_df = data.loc[idxs]


# In[5]:


gid = 2021090900
pid = 97 
nid = 25511 
_df.loc[(_df['gameId']==gid) & (_df['playId']==pid) & (_df['nflId']==nid)]


# In the above example, we can see there are only 6 rows for a player on a given play. This would be the ball snap row and 500ms after ball snap. 

# In[6]:


# get line of scrimmage info to compute block/rush depth relative to LOS
_los = (data
        .loc[(data['team']=='football') & 
             (data['frameId']==1), 
             ['gameId', 'playId', 'x']]
        .rename(columns={'x':'los'}))

# merge LOS info back to subsetted data
_df = _df.merge(_los)


# In[7]:


_df.loc[(_df['gameId']==gid) & (_df['playId']==pid) & (_df['nflId']==nid)]


# The above cells demonstrate taking the line of scrimmage information from the `x` location of the football in the first frame of the play. Alternatively, you could use the `plays.csv` dataset, under the column `absoluteYardlineNumber` which should be the line of scrimmage information as well. 
# 
# Using the same game-play-player example from before: if you scroll to the right, the last column in the dataframe is `los`, which stands for line of scrimmage. 

# In[8]:


# get difference from LOS for all frames and players 
_df['los_diff'] = _df['x'].sub(_df['los'])

# multiply by -1 for plays going the "left" direction 
# this is so pass block is monotonic in the same direction (as well as pass rush)
_df.loc[_df['playDirection']=='left', 'los_diff'] = _df.loc[_df['playDirection']=='left', 'los_diff'].mul(-1)

# merge onto play info to get possession team (could do this anywhere, i do it here for no real optimal reason)
_df = plays.loc[:, ['gameId', 'playId', 'possessionTeam']].merge(_df)


# In[9]:


_df.loc[(_df['gameId']==gid) & (_df['playId']==pid) & (_df['nflId']==nid)]


# We create a difference from line of scrimmage metric `los_diff` and also make sure offense players are going to the "right" and defense players are going to the "left" using the `playDirection` feature and multiplying through by `-1` (arbitrary whether offense or defense is all going left or right; so long as they are going the same way for all rows).  
# 
# In the example we use yet again, we can see we've merged some play data information (possesion team) and the last column in the dataset is the `los_diff` feature. 

# In[10]:


# indicate if a player is on the possession team (1), the defensive team (0), or neither aka the football (-1)
_df['posTeam'] = 0
_df.loc[_df['possessionTeam']==_df['team'], 'posTeam'] = 1 
_df.loc[_df['team']=='football', 'posTeam'] = -1

# create initial snap speed dataframe (mean of speed and acceleration per player)
snap_speed = (_df
              .loc[:, ['nflId','s','a']]
              .groupby('nflId', 
                       as_index=False)
              .mean())


# In[11]:


snap_speed.head()


# We take our temporary dataframe we have been working with and mean aggregate the speed and acceleration data using a groupby method. We get `nflId` and average `s` and `a` as the initial `snap_speed` dataframe. 

# In[12]:


# given whether a offense player or defense player, aggregate by maxmimum or minimum LOS difference, respectively. 
# e.g. if o-lineman has more a negative LOS diff, they allow more pass rush penetration 
_off = _df.loc[_df['posTeam']==1, ['gameId', 'playId', 'nflId', 'los_diff']].groupby(['gameId', 'playId', 'nflId'], as_index=False).max()
_def = _df.loc[_df['posTeam']==0, ['gameId', 'playId', 'nflId', 'los_diff']].groupby(['gameId', 'playId', 'nflId'], as_index=False).min()
los_diff = _off._append(_def)
los_diff = (los_diff
            .loc[:, ['nflId', 'los_diff']]
            .groupby('nflId', 
                     as_index=False)
            .mean())

# merge LOS diff data back onto snap speed
snap_speed = snap_speed.merge(los_diff)
snap_speed = snap_speed.rename(columns={'s':'snap_s', 'a':'snap_a', 'los_diff':'snap_los_diff'})


# In[13]:


snap_speed.head()


# Finally, we aggregate the `los_diff` data and merge that back onto the `snap_speed` dataframe. This gives us new features we can use to analyze player abilities.

# # Exploratory Data Analysis with `snap_speed` data

# In[14]:


df_plt = players.loc[:, ['nflId', 'officialPosition', 'displayName']].merge(snap_speed)
# STEFANOS: Disable plotting
# sns.scatterplot(data=df_plt.loc[df_plt['officialPosition'].isin(['T','G','C','DT','NT','DE'])], x='snap_s', y='snap_los_diff', hue='officialPosition')
# plt.axhline(0, ls=':', lw=2, c='k')
# plt.legend(bbox_to_anchor=(1.02,1), loc=2)
# sns.despine()
# plt.show()


# Using our `snap_speed` dataframe, we can visualize what players are getting over the line of scrimmage faster on defense and what players are blocking near the line of scrimmage. It should be no surprise tackles (who line up off the line of scrimmage) are generally further away from centers (who line up nearly on the line of scrimmage) after the ball is snapped. Some defensive ends seem to be able to get over the line of scrimmage more often than others -- perhaps they are able to time the snap more often (or simply get over the line without getting called offsides). 
# 
# We can also see there is no real correlation with speed of a player on offense. Perhaps there is a correlation on the defensive side (faster defensive players can penetrate deeper beyond the line of scrimmage). 
# 
# Let's take a look at the list of DEs, ordered by line of scrimmage difference

# In[15]:


df_plt.loc[(df_plt['officialPosition']=='DE') & (df_plt['snap_los_diff']<0)].sort_values('snap_los_diff')


# Several of the top players (Bosa, Martin, Ford, etc) seem to be good defensive ends (recent pro bowl/all-pro teams). Perhaps players on this list who are not as well known are underrated or undervalued. 

# # Next steps 
# This is a very simple way to aggregate over the player tracking data in order to create features that can help represent a player's underlying abilities. This was only in relation to half a second after snap -- you can create metrics based around any important moments you define in your dataset. Some examples: 
# * How often do DEs break through being double teamed? 
# * Does a guard get beat more often to his left or to his right? 
# * Does weight/height correlate with overall distance traveled after first contact?  
# 
# Also, remember this is only for one week's worth of data -- it would make sense to loop over all 8 weeks and aggregate all 8 weeks if you want something more comprehensive! 

# # If you liked this notebook, please upvote! 
# # Follow more Big Data Bowl live development on my stream: https://twitch.tv/nickwan_datasci 
