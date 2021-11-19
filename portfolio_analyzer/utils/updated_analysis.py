# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
#Import modules

import pandas as pd
import requests
import numpy as np
import json
from dotenv import load_dotenv
import sys
import os
from alpaca_trade_api.rest import TimeFrame, URL
import alpaca_trade_api as tradeapi
import pytz
import datetime as dt
import holoviews as hv
import panel as pn
import MonteCarloFunctions as mcf
import AlpacaFunctions as apf
import datetime as dt
import hvplot
import hvplot.pandas
import matplotlib.pyplot as plt
import plotly as pty
import plotly.express as px
import bokeh
import streamlit as st


def runUpdatedAnalysis(new_holdings_df, initial_filtered_bar, comparison_std_barplot, combined_sharpe_plot, stacked_bars_plot, initial_portfolio_cum_plot, initial_combined_median_plot, portfolio_intial_distribution_plot):
    # %%
    #Establish ARK API variables -- base url for api calls, request type i.e. profile, trades, etc., etf_symbol for desired etf and additional arguments as parameters
    
    #holdings_symbol = 'ARKK'
    #holdings_url = 'https://arkfunds.io/api/v2/etf/holdings'  

    #Initial API call to establish current positions for ARKK
    # need to code for an error response if API call is unsuccessfsul i.e. i.e.response.status_code == 200:
    #response = requests.get(holdings_url, params = {'symbol' : 'ARKK'}).json()        #print(json.dumps(response, indent=4, sort_keys=True))
    # %% [markdown]
    # **Something for us to consider -- would it be better to utilize dataframes or databases to manipulate and analyze our data?

    
    # %%
    # For our purposes we want to focus on the 'ticker','weight', and 'company' columns of the dataframe.  This will allow us to perform historical research with the Alpaca API as well as plot the weights of the portfolio stocks.
    new_filtered_df = new_holdings_df[['ticker', 'weight', 'company']].sort_values(by = 'weight')
    #st.write(new_filtered_df)

    # Note that for our Monte Carlo simulations, we will need to divide the weights column by 100 since the sum of weights for the simulation needs to be 1, and the dataframe is configured for the sum to be 100.

    new_filtered_bar = new_filtered_df.hvplot.bar(x='ticker', y = 'weight', hover_color = 'red', rot=90, title = 'Stock tickers and their corresponding weights in the updated portfolio', color='green')

    #st.bokeh_chart(hv.render(new_filtered_bar, backend='bokeh'))
    

    #display(initial_filtered_df)


    # %%
    #Use data from ARKK API call to get historical quotes from Alpaca
    new_tickers = new_filtered_df['ticker'].astype(str).tolist()

    timeframe = '1D'
    today = pd.Timestamp.now(tz="America/New_York")
    three_years_ago = pd.Timestamp(today - pd.Timedelta(days=1095)).isoformat()
    end_date = today
    # Do we want to use 1, 2, 3 years of historical data or more? -- 3 for now
    start_date = three_years_ago
    # Here we are retrieving the historical data for the stocks in our new portfolio 
    # We then filter the results to leave us with closing price and ticker columns with a datetime index 
    # so we can run our analyses.

    #ARKK broken up into individual stocks:
    new_portfolio_df = apf.get_historical_dataframe(new_tickers, start_date, end_date, timeframe)
    #st.write(new_portfolio_df)
<<<<<<< HEAD
=======

>>>>>>> a7be19c2d5a86f51dbd4892367d990d47ecfe002


    # Dividing the weights of our new portfolio stocks by 100
    new_mc_weights = list(new_holdings_df.weight / 100)
    # #display(initial_mc_weights)
<<<<<<< HEAD
    
    # Establish # of simulations and number of trading days for our MC Simulations -- hard coded for now,
    # could add functionality to take input in the future
    num_simulations = 50
    num_trading_days = 252 * 2 
=======
    num_simulations = 25
    num_trading_days = 252 
>>>>>>> a7be19c2d5a86f51dbd4892367d990d47ecfe002

    # Creating MC Simulation with our newly updated portfolio
    new_portfolio_sim_input = mcf.configure_monte_carlo(new_portfolio_df, new_mc_weights, num_simulations, num_trading_days)



    # %%
    # We will isolate the daily returns column from our MC_portfolio dataframe in order to run our risk / return analyses by
    # merging the 'daily_return' series from our dataframe 
    new_portfolio_daily_returns = pd.DataFrame()

    # Iterate through tickers in our MC DataFrame, filter to isolate 'daily_return' data, then concat the resulting series into 
    # dataframe we can use for our analyses on the individual stocks in the portfolio
    for ticker in new_tickers:
        daily_returns = new_portfolio_sim_input.portfolio_data[ticker]['daily_return']
        new_portfolio_daily_returns = pd.concat([new_portfolio_daily_returns, daily_returns], axis=1)
    new_portfolio_daily_returns.columns = new_tickers
    



    #Create std plot with new portfolio data
    new_portfolio_std = new_portfolio_daily_returns.std().sort_values()
    new_portfolio_std_barplot = new_portfolio_std.hvplot.bar(title = 'Initial comparisons of standard deviations of daily return data (higher # = more volatile)', color = 'red',
    xlabel = 'Ticker', ylabel = 'Standard Deviation of Daily Returns of New Portfolio Stocks', rot = 90, fontsize = {'title' : '10pt'})
    #st.bokeh_chart(hv.render(new_portfolio_std_barplot, backend='bokeh'))



    # %%
    # Calculating the Sharpe Ratios for our initial portfolio stocks, ARKK and QQQ
    # First we calculate annual return data for our dataset
    year_trading_days = 252
    new_portfolio_annual_return = new_portfolio_daily_returns.mean() * year_trading_days

    # Now we calculate the annualized standard deviation

    new_portfolio_annual_std = new_portfolio_std * np.sqrt(year_trading_days)

    # Lastly, we calculate and plot the Sharpe ratios
    # Calculate the ratios

    new_portfolio_sharpe = new_portfolio_annual_return / new_portfolio_annual_std
    # Plot the ratios

    new_portfolio_sharpe_plot = new_portfolio_sharpe.hvplot.bar(xlabel = 'Tickers', ylabel = 'Sharpe Ratios of new portfolio stocks', title = 'Sharpe Ratios of initial portfolio stocks vs. QQQ and ARKK', color = 'red', label = 'Initial Portfolio Stocks', rot = 90)
    #st.bokeh_chart(hv.render(new_portfolio_sharpe_plot, backend='bokeh'))




    # %%
    # Creating a stacked bar chart for visualization purposes -- values for 'std' and 'sharpe' are multiplied to normalize the values
    #Concatenate the dataframes and format the resulting df
    
    # Establishing index for filtered dataframe for easy concat
    new_filtered_df.index = new_filtered_df.ticker
    
    # Concatenating dataframe to combine sharpe, std and filtered df for our new portfolio
    new_combined_plot_df = pd.concat([new_portfolio_sharpe, new_portfolio_std, new_filtered_df], axis = 1)
    new_combined_plot_df.rename(columns = {0 : 'sharpe', 1 : 'std'}, inplace=True)
    new_combined_plot_df.sort_values(by='std', inplace=True)
    new_combined_plot_df['sharpe'] = new_combined_plot_df['sharpe'] * 3
    new_combined_plot_df['std'] = new_combined_plot_df['std'] * 75
    
    # Creating stacked bars plot and rendering in Streamlit
    new_stacked_bars_plot = new_combined_plot_df.hvplot.bar(rot=90, y=['std', 'sharpe', 'weight'], legend = True, stacked=True, title = 'Sharpe Ratios and Standard Deviations for our new portfolio stocks')
    st.bokeh_chart(hv.render(new_stacked_bars_plot, backend='bokeh'))


    # %%
    # Calculating and plotting cumulative returns for the stocks in our updated portfolio

    new_portfolio_cumprod = (1 + new_portfolio_daily_returns).cumprod()
    new_portfolio_cum_plot = new_portfolio_cumprod.hvplot(kind = 'line', rot=90, title = 'Cumulative returns for the individual stocks in ARKK over time', 
                                                                ylabel = 'Returns', xlabel = 'Date', legend = 'left', fontsize = {'legend' : '8pt'}, frame_height = 250, frame_width = 750)

    # %% [markdown]
    # ### Calling the Monte Carlo functions and running projections -- 

    # Run the simulations:
    new_portfolio_sim_returns = mcf.run_monte_carlo(new_portfolio_sim_input)

    

    new_portfolio_median_outcomes = new_portfolio_sim_returns.median(axis=1)
    new_median_initial_plot = new_portfolio_median_outcomes.hvplot(ylabel = 'Median returns', xlabel = 'Days of projection', title = f'Median returns from {num_simulations} simulations over {num_trading_days} trading days for our new portfolio, QQQ, and ARKK', label = 'New Portfolio')
    
    # Creating overlay plot with our initial ARKK and QQQ plot and then adding in the new portfolio plot
    new_combined_median_plot = new_median_initial_plot * initial_combined_median_plot
    #st.bokeh_chart(hv.render(new_median_initial_plot, backend='bokeh'))
<<<<<<< HEAD
    
    # Rendering to Streamlit
=======
>>>>>>> a7be19c2d5a86f51dbd4892367d990d47ecfe002
    st.bokeh_chart(hv.render(new_combined_median_plot, backend = 'bokeh'))




    # %%
    # Plotting distribrution and confidence intervals from Monte Carlo Simulation
    # This is the plot for the simulations using the individual stocks within ARKK and can be manipulated...
    # this plot will be variable whereas the 'ARKK' and 'QQQ' PLOT

    new_distribution_plot = mcf.plot_distribution(new_portfolio_sim_input)
    st.subheader('Distribution of simulation returns for our updated portfolio')
    st.plotly_chart(new_distribution_plot, sharing="streamlit", title = 'Distribution of cumulative returns across all simulations for the new portfolio')

    
    # %%
    # Describe the MCForecast Summary
    new_portfolio_simulation_summary = mcf.get_monte_summary(new_portfolio_sim_input)
    st.subheader('Summary of returns from the MC simulations on our updated portfolio')
    st.table(new_portfolio_simulation_summary)
    
    




