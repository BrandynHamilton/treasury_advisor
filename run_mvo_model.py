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
#from scripts.utils import 
from scripts.mvo_model import mvo_model
from scripts.data_processing import mvo_combined_assets, combined_all_assets, panama_dao_start_date, current_risk_free, pivot_data_filtered
from scripts.utils import calculate_sortino_ratio

random.seed(42)
np.random.seed(42)

def main():
    rebalancing_frequency = 1
    threshold = 0
    model = mvo_model(current_risk_free, panama_dao_start_date, threshold)
    rebalanced_data = model.rebalance(combined_all_assets, mvo_combined_assets, rebalancing_frequency)
    mvo_daily_portfolio_returns = model.calculate_daily_portfolio_returns(rebalanced_data, mvo_combined_assets)
    cumulative_return = model.calculate_cumulative_return(mvo_daily_portfolio_returns)
    
    # Print the first few rows of the rebalanced data and cumulative return to verify
    print(rebalanced_data.head())
    print(rebalanced_data.tail())
    
    # Prepare base return
    base_return = cumulative_return.reset_index()
    base_return.columns = ['DAY', 'base_cumulative_return']
    
    # Normalize returns
    first_value = base_return['base_cumulative_return'].iloc[0]
    base_return['PanamaDAO_treasury_return'] = 100 + (100 * (base_return['base_cumulative_return'] - first_value))
    
    # Final output
    normalized_returns = base_return[['DAY', 'PanamaDAO_treasury_return']]
    normalized_returns.set_index('DAY', inplace=True)
    
    # Print the first few rows of normalized returns
    print(normalized_returns.head())
    
    # Additional debug plots
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_return, label='Cumulative Return')
    plt.title('Cumulative Return')
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.plot(normalized_returns, label='Normalized Returns')
    plt.title('Normalized Returns')
    plt.xlabel('Date')
    plt.ylabel('Normalized Return')
    plt.legend()
    plt.show()
    
    # Replace values below the threshold with 0
    threshold = 1e-5
    cleaned_rebalanced_data = rebalanced_data.applymap(lambda x: 0 if abs(x) < threshold else x)
    
    # Filter mvo_daily_portfolio_returns by start date
    mvo_daily_portfolio_returns = mvo_daily_portfolio_returns[mvo_daily_portfolio_returns.index >= panama_dao_start_date]
    
    print("Cumulative Return:\n", cumulative_return.head())
    print("Cumulative Return:\n", cumulative_return.tail())
    
    mvo_cumulative_return = cumulative_return.copy()
    
    print("Normalized Returns:\n", normalized_returns.head())
    print("Normalized Returns:\n", normalized_returns.tail())
    mvo_sortino = calculate_sortino_ratio(mvo_daily_portfolio_returns, current_risk_free)

    print(f'mvo sortino ratio: {mvo_sortino}')
    
    mvo_normalized_returns = normalized_returns.copy()
    
    composition_columns = [f'COMPOSITION_{asset}' for asset in mvo_combined_assets]
    mvo_comp = rebalanced_data[composition_columns]
    
    # Replace values below the threshold with 0
    mvo_comp = mvo_comp.applymap(lambda x: 0 if abs(x) < threshold else x)
    
    # Plot the stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    mvo_comp.plot(kind='bar', stacked=True, ax=ax)
    
    # Improve the x-axis labels
    plt.title('MVO Portfolio Composition Over Time')
    plt.xlabel('Date')
    plt.ylabel('Composition')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xticks(ticks=range(0, len(mvo_comp), 30), rotation=45)  # Show x-axis labels every 30 days
    plt.tight_layout()
    plt.show()
    
    latest_comp_mvo = mvo_comp.iloc[-1]
    plt.figure(figsize=(10, 7))
    latest_comp_mvo.plot.pie(autopct='%1.1f%%', startangle=90)
    plt.title('Portfolio Composition')
    plt.ylabel('')  # Hide the y-label
    plt.show()
    
    comp_columns = [col for col in pivot_data_filtered.columns if col.startswith('COMPOSITION_')]
    latest_historical_comp = pivot_data_filtered[comp_columns].iloc[-2]
    plt.figure(figsize=(10, 7))
    latest_historical_comp.plot.pie(autopct='%1.1f%%', startangle=90)
    plt.title('Historical Portfolio Composition')
    plt.ylabel('')  # Hide the y-label
    plt.show()

if __name__ == "__main__":
    main()
