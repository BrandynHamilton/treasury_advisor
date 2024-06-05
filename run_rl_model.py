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

from scripts.rl_model import PortfolioEnv, train_rl_agent
from scripts.utils import calculate_sortino_ratio
from scripts.data_processing import mvo_combined_assets, combined_all_assets, panama_dao_start_date, current_risk_free

random.seed(42)
np.random.seed(42)

def main():
    # TRY 3, 7
    rl_combined_all_assets = combined_all_assets.fillna(0)
    rebalancing_frequency = 5
    
    start_date = panama_dao_start_date
    
    actions_df, rl_portfolio_returns, rl_cumulative_return, rl_normalized_returns, composition_df, returns_df = train_rl_agent(rl_combined_all_assets, mvo_combined_assets, current_risk_free, rebalancing_frequency, start_date)
    
    print(rl_normalized_returns.head())
    print(returns_df.head())
    
    
    # In[86]:
    
    
    composition_df.describe()
    
    
    # In[87]:
    
    
    returns_df
    
    
    # In[88]:
    
    
    actions_df
    
    
    # In[89]:
    
    
    rl_portfolio_returns
    
    
    # In[90]:
    
    
    actions_df['WEIGHT_ETH']
    
    
    # In[91]:
    
    
    composition_df['COMPOSITION_WETH']
    
    
    # In[92]:
    
    
    rl_cumulative_return
    
    
    # In[93]:
    
    
    rl_normalized_returns.set_index('DAY', inplace=True)
    rl_normalized_returns
    
    
    # In[94]:
    
    
    # Find the index of the highest value in the Series
    max_index = rl_portfolio_returns.idxmax()
    
    print("Index of the highest value:", max_index)
    print("Highest value:", rl_portfolio_returns.loc[max_index])
    
    
    # In[95]:
    
    
    composition_df
    
    
    # In[96]:
    
    
    max_index
    
    
    # In[97]:
    
    
    composition_df.index
    
    
    # ## Highest Return Portfolio
    
    # In[98]:
    
    
    composition_df.loc[max_index].sum()
    highest_comp_mvo = composition_df.loc[max_index].iloc[0]
    # Plot the pie chart
    
    
    plt.figure(figsize=(10, 7))
    highest_comp_mvo.plot.pie(autopct='%1.1f%%', startangle=90)
    plt.title('Portfolio Composition')
    plt.ylabel('')  # Hide the y-label
    plt.show()
    
    
    
    # In[99]:
    
    
    rl_normalized_returns.plot()
    
    
    # In[100]:
    
    
    latest_comp_rl = composition_df.iloc[-1]
    # Plot the pie chart
    def rl_composition():
        plt.figure(figsize=(10, 7))
        latest_comp_rl.plot.pie(autopct='%1.1f%%', startangle=90)
        plt.title('Portfolio Composition')
        plt.ylabel('')  # Hide the y-label
        plt.show()
    
    
    # In[175]:
    
    
    import plotly.graph_objs as go
    import plotly.offline as pyo
    import plotly.colors as pc
    
    # Create an empty list to hold the traces
    traces = []
    
    # Define the color palette
    color_palette = pc.qualitative.Plotly
    
    # Loop through each column in composition_df and create a trace for it
    for i, column in enumerate(composition_df.columns):
        trace = go.Bar(
            x=composition_df.index,
            y=composition_df[column],
            name=column,
            marker=dict(color=color_palette[i % len(color_palette)])
        )
        traces.append(trace)
    
    # Create the layout
    layout = go.Layout(
        title='RL Portfolio Composition Over Time',
        barmode='stack',
        xaxis=dict(
            title='Date',
            tickmode='auto',
            nticks=20,
            tickangle=-45
        ),
        yaxis=dict(title='Composition'),
        legend=dict(x=1.05, y=1)
    )
    
    # Combine the data and layout into a figure
    fig = go.Figure(data=traces, layout=layout)
    
    # Render the figure
    pyo.iplot(fig)
    
    # Save the plot as an HTML file
    #pyo.plot(fig, filename='portfolio_composition_over_time.html')
    
    
    # In[101]:
    
    
    rl_portfolio_returns.max()
    
    
    # In[102]:
    
    
    rl_portfolio_returns[rl_portfolio_returns.index >= panama_dao_start_date]
    
    
    # In[103]:
    
    
    rl_sortio = calculate_sortino_ratio(rl_portfolio_returns['Portfolio Return'].values, current_risk_free)
    print(f'rl sortino ratio {rl_sortio}')
    
    
    # In[104]:
    
    
    # Plotting the portfolio value and cumulative return
    
    plt.figure(figsize=(14, 8))
    plt.plot(rl_portfolio_returns.index, rl_portfolio_returns, label='Portfolio Value')
    plt.plot(rl_cumulative_return.index, rl_cumulative_return, label='Cumulative Return')
    plt.title('Portfolio Value and Cumulative Return Over Time')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()
    
    
    # In[105]:
    
    
    # Assuming `cumulative_return` is a Series with index as `DAY`
    
    # Convert cumulative_return Series to a DataFrame and reset index to get `DAY` as a column
    cumulative_return_df = rl_cumulative_return.reset_index()
    cumulative_return_df.columns = ['DAY', 'cumulative_return']
    
    # Prepare base return (use cumulative_return as the base)
    base_return = cumulative_return_df.copy()
    base_return = base_return.dropna().rename(columns={'cumulative_return': 'base_cumulative_return'})
    
    # Combine results
    combined = base_return[['DAY', 'base_cumulative_return']].sort_values('DAY')
    
    # Normalize returns
    first_value = combined['base_cumulative_return'].iloc[0]
    combined['PanamaDAO_treasury_return'] = 100 + (100 * (combined['base_cumulative_return'] - first_value))
    
    # Final output
    rl_normalized_returns = combined[['DAY', 'PanamaDAO_treasury_return']]
    
    # Print the first few rows of normalized returns
    print(rl_normalized_returns.head())
    print(rl_normalized_returns.tail())
    
    
    # In[106]:
    
    
    rl_normalized_returns.set_index('DAY', inplace=True)
    rl_normalized_returns.plot()

if __name__ == "__main__":
    main()
