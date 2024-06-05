import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from scipy.optimize import minimize, Bounds, LinearConstraint
import plotly.graph_objs as go

#print(torch.__version__)
print(PPO)
print(gym.__version__)


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

from scripts.utils import calculate_sortino_ratio, mvo, calculate_log_returns

# ## Trying all assets, historical sortinos

# In[78]:

"""
rl_combined_all_assets = combined_all_assets.fillna(0)


# In[79]:


mvo_combined_assets


# In[80]:


rl_combined_all_assets.iloc[0][[f'COMPOSITION_{asset}' for asset in combined_assets]]


# In[81]:


rl_combined_all_assets.iloc[0][[f'COMPOSITION_{asset}' for asset in all_assets]].values


# In[82]:


prices_input = rl_combined_all_assets[[col for col in rl_combined_all_assets.columns if 'DAILY_PRICE_' in col]]
prices_input


# In[83]:


combined_assets


# ## Data Cleaning

# In[84]:


# Load the prices data
prices_input = rl_combined_all_assets[[col for col in rl_combined_all_assets.columns if 'DAILY_PRICE_' in col]]

# Calculate daily returns
returns = prices_input.pct_change()

# Describe the returns to understand the distribution
returns_description = returns.describe()
print(returns_description)

# Cap extreme values using a threshold based on standard deviation
def cap_extreme_values(series, threshold=3):
    mean = series.mean()
    std = series.std()
    upper_limit = mean + threshold * std
    lower_limit = mean - threshold * std
    return np.clip(series, lower_limit, upper_limit)

# Apply capping to all columns
capped_returns = returns.apply(cap_extreme_values, axis=0)

# Fill missing values by forward filling
cleaned_returns = capped_returns.fillna(method='ffill')

# Verify there are no NaN values left
#assert not cleaned_returns.isnull().values.any(), "There are still NaN values in the cleaned returns."

# Reconstruct the cleaned prices from the cleaned returns
cleaned_prices = (cleaned_returns + 1).cumprod() * prices_input.iloc[0]

# Verify the cleaned prices
cleaned_prices_description = cleaned_prices.describe()
print(cleaned_prices_description)

# Update the original data with cleaned prices
rl_combined_all_assets_cleaned = rl_combined_all_assets.copy()
for col in prices_input.columns:
    rl_combined_all_assets_cleaned[col] = cleaned_prices[col]


# ## Full Log Returns

# In[85]:
"""
def normalize_portfolio(portfolio):
    total = np.sum(portfolio)
    if total != 0:
        return portfolio / total
    return portfolio

class PortfolioEnv(gym.Env):
    def __init__(self, data, initial_composition, all_assets, risk_free, rebalancing_frequency=7, start_date=None):
        super(PortfolioEnv, self).__init__()

        self.data = data
        self.risk_free = risk_free
        self.all_assets = all_assets
        self.num_assets = len(all_assets)
        self.initial_composition = initial_composition
        self.rebalancing_frequency = rebalancing_frequency
        self.start_date = start_date
        self.eth_index = np.where(self.all_assets == 'ETH')[0][0]

        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(2 * self.num_assets,), dtype=np.float32
        )

        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_assets,), dtype=np.float32)

        self.actions_log = []
        self.returns_log = []
        self.composition_log = []
        self.reset()

    def reset(self):
        if self.start_date:
            self.current_step = self.data.index.get_loc(self.start_date)
        else:
            self.current_step = 0
        self.done = False
        self.portfolio = self.initial_composition
        self.actions_log = []
        self.returns_log = []
        self.composition_log = [(self.data.index[self.current_step], self.portfolio.copy())]
        print(f"Environment reset. Initial composition: {self.portfolio}")
        return self._next_observation()

    def _next_observation(self):
        prices = self.data.iloc[self.current_step][[f'DAILY_PRICE_{asset}' for asset in self.all_assets]].values
        obs = np.concatenate([self.portfolio, prices])
        #print(f'portfolio: {self.portfolio}')
        #print(f'prices: {prices}')
        #print(f'obs: {obs}')
        return obs

    def step(self, action):
        if self.done or self.current_step >= len(self.data) - 1:
            self.done = True
            return self._next_observation(), 0, self.done, {}

        current_prices = self.data.iloc[self.current_step][[f'DAILY_PRICE_{asset}' for asset in self.all_assets]].values
        prev_prices = self.data.iloc[self.current_step - 1][[f'DAILY_PRICE_{asset}' for asset in self.all_assets]].values if self.current_step > 0 else current_prices

        #print(f"Step {self.current_step}: Applying daily price changes")
        #print(f"Previous prices: {prev_prices}")
        #print(f"Current prices: {current_prices}")
        #print(f"Portfolio before price change: {self.portfolio}")

        # Calculate log returns
        log_returns = np.log(current_prices / prev_prices)
        #print(f"Log returns: {log_returns}")

        # Apply log returns to the portfolio
        new_portfolio_values = self.portfolio * np.exp(log_returns)
        sum_new_portfolio_values = np.sum(new_portfolio_values)
        #print('new portfolio value', new_portfolio_values)
        #print('sum of new port values', sum_new_portfolio_values)

        if sum_new_portfolio_values != 0:
            self.portfolio = new_portfolio_values / sum_new_portfolio_values
        else:
            self.portfolio = new_portfolio_values

        self.portfolio = normalize_portfolio(new_portfolio_values)
        #print(f"Portfolio after price change: {self.portfolio}")

        if self.current_step % self.rebalancing_frequency == 0:
            action = np.clip(action, 0, 1)
            action_sum = np.sum(action)
            if action_sum == 0:
                action_sum = 1e-8

            action = action / action_sum
            #print(f"Rebalancing step: {self.current_step}")
            #print(f"Action taken: {action}")
            #print(f"Portfolio before rebalancing: {self.portfolio}")

            self.portfolio = action
            self.actions_log.append((self.data.index[self.current_step], action))
            #print(f"Portfolio after rebalancing: {self.portfolio}")

            if self.portfolio[self.eth_index] < 0.2:
                diff = 0.2 - self.portfolio[self.eth_index]
                remaining_allocation = 1 - 0.2

                self.portfolio[self.eth_index] = 0.2
                other_weights = np.delete(self.portfolio, self.eth_index)
                other_weights = other_weights / np.sum(other_weights) * remaining_allocation
                self.portfolio = np.insert(other_weights, self.eth_index, 0.2)

            #print(f"Portfolio after enforcing ETH constraint: {self.portfolio}")

            self.portfolio = normalize_portfolio(self.portfolio)
            #print(f"Normalized portfolio: {self.portfolio}")

        self.composition_log.append((self.data.index[self.current_step], self.portfolio.copy()))

        portfolio_return = np.sum(log_returns * self.portfolio)
        #print('portfolio return', portfolio_return)

        self.returns_log.append((self.data.index[self.current_step], portfolio_return))

        if len(self.returns_log) > 2:
            sortino_ratio = calculate_sortino_ratio([r for _, r in self.returns_log], self.risk_free)
            #print('returns array', [r for _, r in self.returns_log])
        else:
            sortino_ratio = 0

        optimized_weights, _, composition, _ = mvo(self.data.iloc[:self.current_step + 1], self.all_assets, self.risk_free)
        #print('input for mvo', self.data.iloc[:self.current_step + 1])
        if optimized_weights is not None:
            current_weights = composition.iloc[-1].values
            max_distance = sum(abs(1 - value) for value in optimized_weights)
            distance_penalty = sum(abs(current_weights[i] - optimized_weights[i]) for i in range(len(optimized_weights))) / max_distance if max_distance != 0 else 0
        else:
            distance_penalty = 0

        #print(f"Step: {self.current_step}, Action: {action}, Portfolio Return: {portfolio_return}, Sortino Ratio: {sortino_ratio}, Distance Penalty: {distance_penalty}")
        #reward = portfolio_return + 2 * sortino_ratio - distance_penalty 
        reward = portfolio_return + 2 * sortino_ratio - distance_penalty  
        #print('sortino ratio', sortino_ratio)
        #print('reward', reward)
        obs = self._next_observation()
        self.current_step += 1
        return obs, reward, self.done, {}

def train_rl_agent(data, all_assets, risk_free, rebalancing_frequency=7, start_date=None):
    initial_composition = data.loc[data.index >= start_date][[f'COMPOSITION_{asset}' for asset in all_assets]].to_numpy()[0]
    env = PortfolioEnv(data, initial_composition, all_assets, risk_free, rebalancing_frequency=rebalancing_frequency, start_date=start_date)
    model = PPO('MlpPolicy', env, verbose=1)

    total_timesteps = len(data)
    training_duration_factor = 1000
    total_timesteps *= training_duration_factor

    obs = env.reset()
    for _ in range(total_timesteps):
        action, _ = model.predict(obs)
        action = np.clip(action, 0, 1)
        action_sum = np.sum(action)
        if action_sum == 0:
            action_sum = 1e-8
        action = action / action_sum

        obs, rewards, done, info = env.step(action)
        if done:
            break

    actions_log = env.actions_log
    returns_log = env.returns_log
    composition_log = env.composition_log

    actions_df = pd.DataFrame([x[1] for x in actions_log], columns=[f'WEIGHT_{asset}' for asset in all_assets])
    actions_df.index = [x[0] for x in actions_log]

    composition_df = pd.DataFrame([x[1] for x in composition_log], columns=[f'COMPOSITION_{asset}' for asset in all_assets])
    composition_df.index = [x[0] for x in composition_log]

    returns_df = pd.DataFrame([x[1] for x in returns_log], columns=['Portfolio Return'])
    returns_df.index = [x[0] for x in returns_log]

    prices_df = data.loc[data.index >= start_date, [f'DAILY_PRICE_{asset}' for asset in all_assets]]
    prices_df = prices_df.iloc[:-1]
    log_price_returns = calculate_log_returns(prices_df)
    # log_price_returns = log_price_returns.loc[composition_df.index]
    #print('prices df index', prices_df.index.min(), prices_df.index.max())
    #print('composition df index', composition_df.index.min(), composition_df.index.max())
    composition_df = composition_df.loc[log_price_returns.index]
    portfolio_values = (composition_df.values * log_price_returns.values).sum(axis=1)
    rl_portfolio_returns = returns_df
    rl_cumulative_return = (1 + rl_portfolio_returns).cumprod() - 1

    cumulative_return_df = rl_cumulative_return.reset_index()
    cumulative_return_df.columns = ['DAY', 'cumulative_return']

    base_return = cumulative_return_df.dropna().rename(columns={'cumulative_return': 'base_cumulative_return'})

    combined = base_return[['DAY', 'base_cumulative_return']].sort_values('DAY')

    first_value = combined['base_cumulative_return'].iloc[0]
    combined['PanamaDAO_treasury_return'] = 100 + (100 * (combined['base_cumulative_return'] - first_value))

    rl_normalized_returns = combined[['DAY', 'PanamaDAO_treasury_return']]
    rl_normalized_returns.set_index('DAY', inplace=True)

    return actions_df, rl_portfolio_returns, rl_cumulative_return, rl_normalized_returns, composition_df, returns_df

