import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from scipy.optimize import minimize, Bounds, LinearConstraint
import plotly.graph_objs as go

#print(torch.__version__)
print(PPO)
print(gym.__version__)


# In[2]:


import streamlit as st
import pandas as pd
import requests
import numpy as np
import yfinance as yf
import matplotlib
#get_ipython().run_line_magic('matplotlib', 'inline')
import random
import cvxpy as cp
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from scripts.utils import sortino


# In[3]:


import sys
print(sys.executable)


# In[4]:


random.seed(42)
np.random.seed(42)


# # Data Cleaning & Models

# ## Historical

# In[5]:


tbill_path = 'data/3_month_t_bill.csv'
tbill = pd.read_csv(tbill_path)
tbill['decimal'] = tbill['TB3MS'] / 100
current_risk_free = tbill['decimal'].iloc[-1]
current_risk_free


# In[6]:


assets_path = 'data/asset_prices.csv'
assets = pd.read_csv(assets_path)
assets.dropna(inplace=True)
all_assets = assets.copy()
all_assets = all_assets[all_assets['DAY'] >= '2023-01-22']
assets = assets[assets['DAY'] >= '2023-03-22']
assets


# In[7]:


all_assets_pivot = all_assets.pivot(index='DAY', columns='SYMBOL', values='PRICE').reset_index()
all_assets_pivot.set_index('DAY', inplace=True)
all_assets_pivot.columns = [f'DAILY_PRICE_{col}' for col in all_assets_pivot.columns]
all_assets_pivot


# In[8]:


all_assets_pivot.describe()


# In[9]:


# Assuming 'assets' data is loaded into a DataFrame called assets_df

# Convert 'DAY' to datetime
assets['DAY'] = pd.to_datetime(assets['DAY'])

# Pivot the data to have symbols as columns
pivot_assets = assets.pivot(index='DAY', columns='SYMBOL', values='PRICE').reset_index()

# Display the pivoted DataFrame
print(pivot_assets.head())
pivot_assets.set_index('DAY', inplace=True)


# In[10]:


pivot_assets.columns = [f'DAILY_PRICE_{col}' for col in pivot_assets.columns]
pivot_assets = pivot_assets.reset_index()
pivot_assets


# In[11]:


panamadao_path = 'data/panamadao_returns.csv'
panamadao_returns = pd.read_csv(panamadao_path)
panamadao_returns.dropna(inplace=True)
panamadao_returns


# In[12]:


panama_dao_assets = panamadao_returns['SYMBOL'].unique()


# # Convert 'DAY' to datetime
# panamadao_returns['DAY'] = pd.to_datetime(panamadao_returns['DAY'])
# 
# # Sort by 'DAY' and 'SYMBOL'
# panamadao_returns = panamadao_returns.sort_values(by=['DAY', 'SYMBOL'])
# 
# # Calculate previous day price
# panamadao_returns['prev_day_price'] = panamadao_returns.groupby('SYMBOL')['DAILY_PRICE'].shift(1)
# 
# # Calculate daily return
# panamadao_returns['daily_return'] = (panamadao_returns['DAILY_PRICE'] - panamadao_returns['prev_day_price']) / panamadao_returns['prev_day_price']
# 
# # Calculate weighted daily return
# panamadao_returns['weighted_daily_return1'] = panamadao_returns['daily_return'] * panamadao_returns['COMPOSITION']
# 
# # Filter dates after '2023-07-01'
# filtered_returns = panamadao_returns
# 
# # Calculate weighted daily return per day
# weighted_daily_returns = filtered_returns.groupby('DAY').apply(lambda x: (x['daily_return'] * x['COMPOSITION']).sum()).reset_index(name='weighted_daily_return')
# 
# # Calculate cumulative returns
# weighted_daily_returns['daily_factor'] = 1 + weighted_daily_returns['weighted_daily_return']
# weighted_daily_returns['cumulative_return'] = (np.exp(np.log(weighted_daily_returns['daily_factor']).cumsum()) - 1)
# 
# # Calculate metrics
# average_daily_return = weighted_daily_returns['weighted_daily_return'].mean()
# stddev_daily_return = weighted_daily_returns['weighted_daily_return'].std()
# 
# # Calculate Sharpe ratio
# annualized_return = average_daily_return * 365
# annualized_risk = stddev_daily_return * np.sqrt(365)
# sharpe_ratio = (annualized_return - current_risk_free) / annualized_risk
# 
# # Prepare base return
# base_return = weighted_daily_returns[['DAY', 'cumulative_return']].copy()
# base_return = base_return.dropna().rename(columns={'cumulative_return': 'base_cumulative_return'})
# 
# # Combine results
# combined = base_return[['DAY', 'base_cumulative_return']].sort_values('DAY')
# 
# # Normalize returns
# first_value = combined['base_cumulative_return'].iloc[0]
# combined['PanamaDAO_treasury_return'] = 100 + (100 * (combined['base_cumulative_return'] - first_value))
# 
# # Final output
# historical_normalized_returns = combined[['DAY', 'PanamaDAO_treasury_return']]
# 
# print(historical_normalized_returns)

# In[13]:


# Convert 'DAY' to datetime
panamadao_returns['DAY'] = pd.to_datetime(panamadao_returns['DAY'])

# Sort by 'DAY' and 'SYMBOL'
panamadao_returns = panamadao_returns.sort_values(by=['DAY', 'SYMBOL'])

# Calculate previous day price
panamadao_returns['prev_day_price'] = panamadao_returns.groupby('SYMBOL')['DAILY_PRICE'].shift(1)

# Calculate daily log return
panamadao_returns['daily_log_return'] = np.log(panamadao_returns['DAILY_PRICE'] / panamadao_returns['prev_day_price'])

# Calculate weighted daily log return
panamadao_returns['weighted_daily_return'] = panamadao_returns['daily_log_return'] * panamadao_returns['COMPOSITION']

# Filter dates after '2023-07-01'
filtered_returns = panamadao_returns

# Calculate weighted daily log return per day
weighted_daily_log_returns = filtered_returns.groupby('DAY').apply(lambda x: x['weighted_daily_return'].sum()).reset_index(name='weighted_daily_return')

# Calculate cumulative log returns
weighted_daily_log_returns['cumulative_log_return'] = weighted_daily_log_returns['weighted_daily_return'].cumsum()

# Calculate cumulative returns from cumulative log returns
weighted_daily_log_returns['cumulative_return'] = np.exp(weighted_daily_log_returns['cumulative_log_return']) - 1



# Calculate metrics
average_daily_log_return = weighted_daily_log_returns['weighted_daily_return'].mean()
stddev_daily_log_return = weighted_daily_log_returns['weighted_daily_return'].std()

# Calculate Sharpe ratio
annualized_return = average_daily_log_return * 365
annualized_risk = stddev_daily_log_return * np.sqrt(365)
sharpe_ratio = (annualized_return - current_risk_free) / annualized_risk

# Prepare base return
base_return = weighted_daily_log_returns[['DAY', 'cumulative_return']].copy()
base_return = base_return.dropna().rename(columns={'cumulative_return': 'base_cumulative_return'})

# Combine results
combined = base_return[['DAY', 'base_cumulative_return']].sort_values('DAY')

# Normalize returns
first_value = combined['base_cumulative_return'].iloc[0]
combined['PanamaDAO_treasury_return'] = 100 + (100 * (combined['base_cumulative_return'] - first_value))

# Final output
historical_normalized_returns = combined[['DAY', 'PanamaDAO_treasury_return']]

print(historical_normalized_returns)


# In[14]:


weighted_daily_returns = weighted_daily_log_returns


# In[15]:


historical_normalized_returns.set_index('DAY', inplace=True)


# In[16]:


#historical_normalized_returns.plot()


# In[17]:


panamadao_returns


# In[18]:


pivot_data = panamadao_returns.pivot(index='DAY', columns='SYMBOL', values=['TOTAL_FILLED_BALANCE', 'DAILY_PRICE', 'TOTAL_VALUE_IN_USD', 'COMPOSITION'])
pivot_data.columns = ['_'.join(col).strip() for col in pivot_data.columns.values]
pivot_data = pivot_data.reset_index()


# In[19]:


pivot_data.set_index('DAY', inplace=True)
pivot_data


# In[20]:


filtered_columns = [col for col in pivot_data.columns if 'COMPOSITION_' in col or 'DAILY_PRICE_' in col]
pivot_data_filtered = pivot_data[filtered_columns]
pivot_data_filtered


# In[21]:


# Convert assets DataFrame to wide format
assets_pivot = assets.pivot(index='DAY', columns='SYMBOL', values='PRICE')

# Rename columns to match the format in pivot_data_filtered
assets_pivot.columns = [f'DAILY_PRICE_{col}' for col in assets_pivot.columns]
assets_pivot.index


# In[22]:


assets_pivot


# In[23]:


pivot_data_filtered.index = pd.to_datetime(pivot_data_filtered.index)


# In[24]:


all_assets_pivot.index = pd.to_datetime(all_assets_pivot.index)
all_assets_pivot.index


# In[25]:


pivot_data_filtered.index


# In[26]:


pivot_data_filtered.columns


# In[27]:


pivot_data_filtered_no_price = pivot_data_filtered.drop(columns=['DAILY_PRICE_ETH', 'DAILY_PRICE_USDC', 'DAILY_PRICE_WETH'])

combined_all_assets = all_assets_pivot.join(pivot_data_filtered_no_price,  how='left')
combined_all_assets = combined_all_assets[combined_all_assets.index <= '2024-05-15']
combined_all_assets


# In[28]:


all_assets_pivot.index


# In[29]:


pivot_data_filtered_no_price.index


# In[30]:


combined_all_assets


# In[31]:


assets_pivot_smaller_assets = assets_pivot.drop(columns=['DAILY_PRICE_USDC','DAILY_PRICE_WETH'])


# In[32]:


# Merge the two DataFrames on the 'DAY' index
combined_data1 = pivot_data_filtered.merge(assets_pivot_smaller_assets, left_index=True, right_index=True, how='left')

combined_data1



# In[33]:


for asset in assets['SYMBOL'].unique():
    comp_col = f'COMPOSITION_{asset}'
    if comp_col not in combined_all_assets.columns:
        combined_all_assets[comp_col] = 0.0


# In[34]:


# Initialize the composition columns for new assets with zeros
for asset in assets['SYMBOL'].unique():
    comp_col = f'COMPOSITION_{asset}'
    if comp_col not in combined_data1.columns:
        combined_data1[comp_col] = 0.0

combined_data1


# In[35]:


all_assets = ['ETH', 'USDC', 'WETH', 'DPI', 'PAXG', 'RETH', 'SOL', 'WBTC', 'WSTETH']

# Construct the column names for compositions and daily prices
composition_columns = [f'COMPOSITION_{asset}' for asset in all_assets]
price_columns = [f'DAILY_PRICE_{asset}' for asset in all_assets]

# Combine the columns to filter
required_columns = composition_columns + price_columns

# Filter the combined_data1 DataFrame for the required columns
filtered_combined_data1 = combined_data1[required_columns]

# Print the first few rows to verify
print(filtered_combined_data1.head())


# In[190]:


#combined_data1.to_csv('data/combined_data.csv')


# In[37]:


weighted_daily_returns


# In[38]:


current_risk_free


# In[39]:


weighted_daily_returns.set_index('DAY', inplace=True)


# In[40]:


historical_returns = weighted_daily_returns['weighted_daily_return']



sortino(historical_returns, current_risk_free)


# In[41]:


historical_cumulative_return = weighted_daily_returns['cumulative_return']
#historical_cumulative_return.plot()


# In[42]:


combined_data = pivot_data.merge(pivot_assets, on='DAY', how='left')
combined_data.set_index('DAY', inplace=True)
combined_data


# In[43]:


portfolio = combined_data[[col for col in combined_data.columns if 'COMPOSITION_' in col]]
asset_prices = combined_data[[col for col in combined_data.columns if 'DAILY_PRICE_' in col]]
print(portfolio)
print(asset_prices)


# ## MVO Rebalancing

# In[44]:


panama_dao_start_date = panamadao_returns['DAY'].iloc[0]
panama_dao_start_date


# In[45]:


combined_all_assets


# In[46]:


combined_all_assets['COMPOSITION_ETH']


# In[47]:


all_assets = assets['SYMBOL'].unique()

start_data = combined_all_assets[combined_all_assets.index >= panama_dao_start_date]
composition_columns = [f'COMPOSITION_{asset}' for asset in all_assets]
starting_composition = start_data[composition_columns]


# In[48]:


pivot_data_filtered.columns


# In[49]:


panama_dao_assets


# In[50]:


full_starting_composition = starting_composition.merge(pivot_data_filtered['COMPOSITION_ETH'], left_index=True, right_index=True)


# In[51]:


full_starting_composition[['COMPOSITION_ETH', 'COMPOSITION_USDC', 'COMPOSITION_WETH']]


# In[52]:


print('all assets', all_assets)
print('panamadao assets', panama_dao_assets)

# Combine both arrays using np.concatenate and remove duplicates using np.unique
combined_assets = np.unique(np.concatenate((all_assets, panama_dao_assets)))

print('combined assets', combined_assets)
print('panamadao assets', panama_dao_assets)


# In[53]:


combined_all_assets['DAILY_PRICE_USDC'] = combined_all_assets['DAILY_PRICE_USDC'].clip(upper=1.15).apply(lambda x: 1 if x > 1 else x)


# In[54]:


combined_all_assets['DAILY_PRICE_USDC'].describe()


# In[55]:


combined_all_assets['DAILY_PRICE_ETH'] = combined_all_assets['DAILY_PRICE_WETH']


# In[56]:


prices_all_assets = combined_all_assets[[f'DAILY_PRICE_{asset}' for asset in combined_assets]]
prices_all_assets.pct_change().describe()


# In[57]:


assets_to_drop = ['G', 'HOP','CELR','CDAI','CETH','REN','STG','AAVE','FRAX','MPL','RSR','BAL','ARB','ENS']
mvo_combined_assets = np.array([asset for asset in combined_assets if asset not in assets_to_drop])

#mvo_combined_assets


# In[58]:


iniital_composition = full_starting_composition.iloc[0]
latest_composition = full_starting_composition.iloc[-1]


# In[59]:


#latest_composition.plot.pie(figsize=(10, 10), autopct='%1.1f%%')


# In[60]:


#iniital_composition.plot.pie(figsize=(10, 10), autopct='%1.1f%%')


# from notebook.services.config import ConfigManager
# cm = ConfigManager()
# cm.update('notebook', {
#     'NotebookApp': {
#         'iopub_data_rate_limit': 10000000
#     }
# })
# 

# In[61]:


starting_row = combined_all_assets[combined_all_assets.index >= panama_dao_start_date].iloc[0]


# In[62]:


starting_row[[f'COMPOSITION_{asset}' for asset in mvo_combined_assets]]

indicies_path = 'data/index.csv'
indicies = pd.read_csv(indicies_path)
indicies.set_index('DAY', inplace=True)
indicies.dropna(inplace=True)
index_start = indicies.index[-1]
index_start
indicies