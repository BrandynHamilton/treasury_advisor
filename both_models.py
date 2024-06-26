import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from scipy.optimize import minimize, Bounds, LinearConstraint

#print(torch.__version__)
print(PPO)
print(gym.__version__)


# In[2]:
import plotly.graph_objs as go
import plotly.offline as pyo
import plotly.colors as pc
import plotly.subplots as sp


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

from scripts.utils import calculate_sortino_ratio, calculate_treynor_ratio, calculate_cagr, calculate_beta, normalize
from scripts.data_processing import indicies, index_start, combined_all_assets, mvo_combined_assets, panama_dao_start_date, current_risk_free, historical_returns, historical_cumulative_return, historical_normalized_returns, pivot_data_filtered, pivot_assets
from scripts.rl_model import PortfolioEnv, train_rl_agent
from scripts.mvo_model import mvo_model


combined_all_assets = combined_all_assets.fillna(0)
rl_rebalancing_frequency = 5

start_date = panama_dao_start_date

actions_df, rl_portfolio_returns, rl_cumulative_return, rl_normalized_returns, composition_df, returns_df = train_rl_agent(combined_all_assets, mvo_combined_assets, current_risk_free, rl_rebalancing_frequency, start_date)


mvo_rebalancing_frequency = 1
threshold = 0
model = mvo_model(current_risk_free, panama_dao_start_date, threshold)
rebalanced_data = model.rebalance(combined_all_assets, mvo_combined_assets, mvo_rebalancing_frequency)
mvo_daily_portfolio_returns = model.calculate_daily_portfolio_returns(rebalanced_data, mvo_combined_assets)
mvo_daily_portfolio_returns = mvo_daily_portfolio_returns[mvo_daily_portfolio_returns.index >= panama_dao_start_date]
mvo_cumulative_return = model.calculate_cumulative_return(mvo_daily_portfolio_returns)

base_return = mvo_cumulative_return.reset_index()
base_return.columns = ['DAY', 'base_cumulative_return']

# Normalize returns
first_value = base_return['base_cumulative_return'].iloc[0]
base_return['PanamaDAO_treasury_return'] = 100 + (100 * (base_return['base_cumulative_return'] - first_value))

# Final output
mvo_normalized_returns = base_return[['DAY', 'PanamaDAO_treasury_return']]
mvo_normalized_returns.set_index('DAY', inplace=True)



print(f'rl normalized returns index from {rl_normalized_returns.index.min()} to {rl_normalized_returns.index.max()}')
print(f'mvo normalized returns index from {mvo_normalized_returns.index.min()} to {mvo_normalized_returns.index.max()}')
print(f'historical normalized returns index from {historical_normalized_returns.index.min()} to {historical_normalized_returns.index.max()}')



rl_portfolio_returns


# In[108]:


mvo_daily_portfolio_returns


# In[109]:


historical_returns


# ### Sortino Ratios

# In[110]:





# sortino(historical_returns.values)

# In[187]:


historical_sortino = calculate_sortino_ratio(historical_returns.values, current_risk_free)


# sortino(mvo_daily_portfolio_returns.values)

# In[188]:


mvo_sortino = calculate_sortino_ratio(mvo_daily_portfolio_returns.values, current_risk_free)



# In[113]:


# In[189]:


rl_sortino = calculate_sortino_ratio(rl_portfolio_returns['Portfolio Return'].values, current_risk_free)

print(f'historical sortino: {historical_sortino}')
print(f'mvo sortino: {mvo_sortino}')
print(f'rl sortino: {rl_sortino}')



# ## Cumulative Returns

# In[115]:


rl_cumulative_return


# In[116]:


mvo_cumulative_return


# In[117]:


historical_cumulative_return


# ## Normalized Returns

# In[118]:


rl_normalized_returns


# In[119]:


mvo_normalized_returns


# In[120]:


historical_normalized_returns


# In[121]:





# In[122]:





# In[123]:


rl_cumulative_return


# In[124]:


index_historical_normalized = normalize(historical_cumulative_return, index_start)
index_rl_normalized = normalize(rl_cumulative_return, index_start)
index_mvo_normalized = normalize(mvo_cumulative_return, index_start)


# In[125]:


index_mvo_normalized.rename(columns={'treasury_return':'MVO Treasury Robo Advisor Return'}, inplace=True)


# In[126]:


index_rl_normalized.rename(columns={'treasury_return':'RL Treasury Robo Advisor Return'}, inplace=True)


# In[127]:


index_historical_normalized.rename(columns={'treasury_return':'Historical PanamaDAO Treasury Return'}, inplace=True)


# In[128]:


index_historical_normalized.index


# In[129]:


indicies.index = pd.to_datetime(indicies.index)
combined_indicies = indicies.merge(index_historical_normalized, left_index=True, right_index=True, how='inner')
combined_indicies = combined_indicies.merge(index_mvo_normalized, left_index=True, right_index=True, how='inner')
combined_indicies = combined_indicies.merge(index_rl_normalized, left_index=True, right_index=True, how='inner')


# In[130]:


combined_indicies.columns


# In[131]:


combined_indicies.drop(columns=['WBTC_DAILY_RETURN','WETH_DAILY_RETURN','WBTC_NORMALIZED_PRICE','WETH_NORMALIZED_PRICE'], inplace=True)


# In[132]:





# In[167]:




# Create traces for each line
trace1 = go.Scatter(
    x=historical_normalized_returns.index,
    y=historical_normalized_returns['PanamaDAO_treasury_return'],
    mode='lines',
    name='Historical Normalized Returns',
    line=dict(color='blue')
)

trace2 = go.Scatter(
    x=mvo_normalized_returns.index,
    y=mvo_normalized_returns['PanamaDAO_treasury_return'],
    mode='lines',
    name='MVO Normalized Returns',
    line=dict(color='red')
)

trace3 = go.Scatter(
    x=rl_normalized_returns.index,
    y=rl_normalized_returns['PanamaDAO_treasury_return'],
    mode='lines',
    name='RL Normalized Returns',
    line=dict(color='green')
)

# Create the layout
layout = go.Layout(
    title='Comparison of Historical and Robo Advisor Normalized Returns',
    xaxis=dict(title='Date'),
    yaxis=dict(title='Normalized Returns'),
    legend=dict(x=0.1, y=0.9)
)

# Combine the data and layout into a figure
fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)

# Render the figure
pyo.iplot(fig)

# Save the plot as an HTML file
#pyo.plot(fig, filename='normalized_returns_comparison.html')

# Extract the last values from each cumulative return Series
historical_return = float(historical_cumulative_return.iloc[-1]) * 100
rl_return = float(rl_cumulative_return.iloc[-1]) * 100
mvo_return = float(mvo_cumulative_return.iloc[-1]) * 100

# Convert the returns to strings and print
print(f'historical cumulative return {historical_return:.2f}%')
print(f'rl cumulative return {rl_return:.2f}%')
print(f'mvo cumulative return {mvo_return:.2f}%')


# In[170]:




# Define a color palette
color_palette = pc.qualitative.Plotly

# Create an empty list to hold the traces
traces = []

# Loop through each column in combined_indicies and create a trace for it
for i, column in enumerate(combined_indicies.columns):
    trace = go.Scatter(
        x=combined_indicies.index,
        y=combined_indicies[column],
        mode='lines',
        name=column,
        line=dict(color=color_palette[i % len(color_palette)])
    )
    traces.append(trace)

# Create the layout
layout = go.Layout(
    title='Normalized Comparison of DAO Indicies to Robo Advisor Returns',
    xaxis=dict(title='Date'),
    yaxis=dict(title='Value'),
    legend=dict(x=0.1, y=0.9)
)

# Combine the data and layout into a figure
fig = go.Figure(data=traces, layout=layout)

# Render the figure
pyo.iplot(fig)

# Save the plot as an HTML file
#pyo.plot(fig, filename='dao_indices_vs_robo_advisor_returns.html')


# In[169]:

composition_columns = [f'COMPOSITION_{asset}' for asset in mvo_combined_assets]
mvo_comp = rebalanced_data[composition_columns]
latest_comp_mvo = mvo_comp.iloc[-1]

latest_comp_rl = composition_df.iloc[-1]

comp_columns = [col for col in pivot_data_filtered.columns if col.startswith('COMPOSITION_')]

latest_historical_comp = pivot_data_filtered[comp_columns].iloc[-2]




latest_comp_mvo


# In[136]:


latest_historical_comp


# In[137]:


latest_comp_rl


# In[183]:



# Define a function to create a pie chart trace for MVO composition
def mvo_composition():
    return go.Pie(
        labels=latest_comp_mvo.index,
        values=latest_comp_mvo.values,
        name='MVO Portfolio Composition',
        hole=0.3,
        marker=dict(colors=pc.qualitative.Pastel)
    )

# Define a function to create a pie chart trace for historical composition
def historical_composition():
    return go.Pie(
        labels=latest_historical_comp.index,
        values=latest_historical_comp.values,
        name='Historical Portfolio Composition',
        hole=0.3,
        marker=dict(colors=pc.qualitative.Set3)
    )

# Define a function to create a pie chart trace for RL composition
def rl_composition():
    return go.Pie(
        labels=latest_comp_rl.index,
        values=latest_comp_rl.values,
        name='RL Portfolio Composition',
        hole=0.3,
        marker=dict(colors=pc.qualitative.Alphabet)
    )

# Create subplots
fig = sp.make_subplots(
    rows=1, cols=3,
    specs=[[{'type': 'domain'}, {'type': 'domain'}, {'type': 'domain'}]],
    subplot_titles=['MVO', 'Historical', 'RL']
)

# Add traces to the subplots
fig.add_trace(mvo_composition(), row=1, col=1)
fig.add_trace(historical_composition(), row=1, col=2)
fig.add_trace(rl_composition(), row=1, col=3)

# Update layout
fig.update_layout(
    title_text='Portfolio Compositions',
    title_x=0.5,  # Center the main title
    title_y=0.95,  # Position the main title slightly lower
    title_xanchor='center',  # Anchor the title at the center
    title_yanchor='top'  # Anchor the title at the top
)

# Show the plot
fig.show()

# Save the plot as an HTML file
#fig.write_html('portfolio_compositions.html')


# ## Market Comparisons - Beta

# In[139]:


#!pip install scikit-learn


# In[140]:


# Align the data sets by date
# Prepare data for linear regression




# In[142]:


combined_indicies


# In[143]:


rl_beta = calculate_beta(combined_indicies, 'SIRLOIN_INDEX', 'RL Treasury Robo Advisor Return')
rl_cagr = calculate_cagr(combined_indicies['RL Treasury Robo Advisor Return'])
print(f"RL Beta: {rl_beta}")
print(f"RL CAGR is {rl_cagr * 100:.2f}%")


# In[144]:


mvo_beta = calculate_beta(combined_indicies, 'SIRLOIN_INDEX', 'MVO Treasury Robo Advisor Return')
mvo_cagr = calculate_cagr(combined_indicies['MVO Treasury Robo Advisor Return'])
print(f"MVO Beta: {mvo_beta}")
print(f"MVO CAGR is {mvo_cagr * 100:.2f}%")


# In[145]:


historical_beta = calculate_beta(combined_indicies, 'SIRLOIN_INDEX', 'Historical PanamaDAO Treasury Return')
historical_cagr = calculate_cagr(combined_indicies['Historical PanamaDAO Treasury Return'])
print(f"Historical Beta: {historical_beta}")
print(f"Historical CAGR is {historical_cagr*100:.2f}%")


# In[146]:


prices = pivot_assets.copy()
prices.set_index('DAY', inplace=True)
dpi_history = prices['DAILY_PRICE_DPI']


# In[147]:


dpi_history.head()


# In[148]:


dpi_cagr = calculate_cagr(dpi_history)
    
dpi_cumulative_risk_premium = dpi_cagr - current_risk_free
print('DPI CAGR:', dpi_cagr)
print('DPI Cumulative Risk Premium:',dpi_cumulative_risk_premium)


# In[149]:


#DEFI_TREASURY_INDEX

defi_index_history = combined_indicies['DEFI_TREASURY_INDEX']

defi_index_cagr = calculate_cagr(defi_index_history)
defi_index_cumulative_risk_premium = defi_index_cagr - current_risk_free
print('Defi Index CAGR:', defi_index_cagr)
print('Defi Index Cumulative Risk Premium:',defi_index_cumulative_risk_premium)


# In[150]:


#DEFI_TREASURY_INDEX

sirloin_history = combined_indicies['SIRLOIN_INDEX']

sirloin_cagr = calculate_cagr(sirloin_history)
sirloin_cumulative_risk_premium = sirloin_cagr - current_risk_free
print('Sirlion CAGR:', sirloin_cagr)
print('Sirloin Cumulative Risk Premium:',sirloin_cumulative_risk_premium)


# In[151]:


current_risk_free


# In[152]:


combined_indicies.columns


# In[153]:


defi_index_cagr = calculate_cagr(combined_indicies['DEFI_TREASURY_INDEX'])
non_defi_index_cagr = calculate_cagr(combined_indicies['NON_DEFI_TREASURY_INDEX'])

defi_index_beta = calculate_beta(combined_indicies, 'SIRLOIN_INDEX', 'DEFI_TREASURY_INDEX')
non_defi_index_beta = calculate_beta(combined_indicies, 'SIRLOIN_INDEX', 'NON_DEFI_TREASURY_INDEX')



# In[154]:


defi_index_beta


# In[155]:


non_defi_index_beta


# ## Treynor Ratios

# In[156]:





# In[157]:


historical_treynor = calculate_treynor_ratio(historical_returns.values,historical_beta, current_risk_free)


# In[158]:


mvo_treynor = calculate_treynor_ratio(mvo_daily_portfolio_returns.values,mvo_beta, current_risk_free)


# In[159]:


rl_treynor = calculate_treynor_ratio(rl_portfolio_returns['Portfolio Return'].values,rl_beta, current_risk_free)


# ## SML

# In[160]:


def create_interactive_sml(risk_free_rate, market_risk_premium, rl_beta, mvo_beta, historical_beta, defi_beta, non_defi_beta, rl_return, mvo_return, historical_return, defi_return, non_defi_return):
        betas = np.linspace(0, 6, 100)
        expected_returns = risk_free_rate + betas * market_risk_premium
    
        # Create the SML line
        sml_line = go.Scatter(x=betas, y=expected_returns, mode='lines', name=f'SML')
    
        # Add MakerDAO token as points for expected (based on selected market risk premium) and actual returns
        rl_expected = go.Scatter(
            x=[rl_beta],
            y=[risk_free_rate + rl_beta * market_risk_premium],  # Use the selected market risk premium
            mode='markers', 
            marker=dict(color='red', size=10),
            name=f'RL Expected'
        )
        
        rl_actual = go.Scatter(
            x=[rl_beta],
            y=[rl_return],
            mode='markers', 
            marker=dict(color='pink', size=10),
            name=f'RL Actual'
        )
    
        # Add LidoDAO token as points for expected and actual returns
        mvo_expected = go.Scatter(
            x=[mvo_beta],
            y=[risk_free_rate + mvo_beta * market_risk_premium],  # Use the selected market risk premium
            mode='markers', 
            marker=dict(color='blue', size=10),
            name=f'MVO Expected'
        )
        
        mvo_actual = go.Scatter(
            x=[mvo_beta],
            y=[mvo_return],
            mode='markers', 
            marker=dict(color='lightblue', size=10),
            name=f'MVO Actual'
        )
    
        # Convert rpl_beta from array to scalar if necessary
        #rpl_beta_value = rpl_beta[0] if isinstance(rpl_beta, np.ndarray) else rpl_beta
    
        # Add Rocket Pool token as points for expected and actual returns
        historical_expected = go.Scatter(
            x=[historical_beta],
            y=[risk_free_rate + historical_beta * market_risk_premium],  # Use the selected market risk premium
            mode='markers',
            marker=dict(color='orange', size=10),
            name='Historical Expected'
        )
        
        historical_actual = go.Scatter(
            x=[historical_beta],
            y=[historical_return],
            mode='markers',
            marker=dict(color='yellow', size=10),
            name='Historical Actual'
         
        )
        
        defi_expected = go.Scatter(
            x=[defi_beta],
            y=[risk_free_rate + defi_beta * market_risk_premium],  # Use the selected market risk premium
            mode='markers',
            marker=dict(color='purple', size=10),
            name='Defi Treasury Index Expected'
            
        )
        
        defi_actual = go.Scatter(
            x=[defi_beta],
            y=[defi_return],
            mode='markers',
            marker=dict(color='navy', size=10),
            name='Defi Treasury Index Actual'
        
        )
        
        non_defi_expected = go.Scatter(
            x=[non_defi_beta],
            y=[risk_free_rate + non_defi_beta * market_risk_premium],  # Use the selected market risk premium
            mode='markers',
            marker=dict(color='grey', size=10),
            name='Non-Defi Treasury Index Expected'
      
        )
        
        non_defi_actual = go.Scatter(
            x=[non_defi_beta],
            y=[non_defi_return],
            mode='markers',
            marker=dict(color='tan', size=10),
            name='Non-Defi Treasury Index Actual'
           
        )
        
        
    
        # Add Risk-Free Rate line
        risk_free_line = go.Scatter(
            x=[0, max(betas)], 
            y=[risk_free_rate, risk_free_rate], 
            mode='lines', 
            line=dict(dash='dash', color='green'),
            name='Risk-Free Rate'
        )
    
        # Layout settings
        layout = go.Layout(
            title=f'SML',
            xaxis=dict(title='Beta (Systematic Risk)'),
            yaxis=dict(title='Return'),
            showlegend=True
        )
    
        # Combine all the plots
        fig = go.Figure(data=[sml_line, rl_expected, rl_actual, mvo_expected, mvo_actual, historical_expected, historical_actual, 
                              defi_expected, defi_actual, non_defi_expected,non_defi_actual, risk_free_line], layout=layout)
        return fig, market_risk_premium


# In[161]:


historical_cagr


# In[162]:


fig, market_risk_premium = create_interactive_sml(
    current_risk_free,
    sirloin_cumulative_risk_premium,
    rl_beta,
    mvo_beta,
    historical_beta,
    defi_index_beta,
    non_defi_index_beta,
    rl_cagr,
    mvo_cagr,
    historical_cagr,
    defi_index_cagr,
    non_defi_index_cagr

    

)


# In[163]:


pyo.iplot(fig)
