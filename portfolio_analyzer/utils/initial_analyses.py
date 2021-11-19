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
import MCForecastTools
import MonteCarloFunctions as mcf
import AlpacaFunctions as apf
from updated_analysis import runUpdatedAnalysis
import datetime as dt
import hvplot
import hvplot.pandas
import matplotlib.pyplot as plt
import plotly as pty
import plotly.express as px
import bokeh
import streamlit as st

def runFirstAnalysis():
    # %%
    #Establish ARK API variables -- base url for api calls, request type i.e. profile, trades, etc., etf_symbol for desired etf and additional arguments as parameters
    
    holdings_symbol = 'ARKK'
    holdings_url = 'https://arkfunds.io/api/v2/etf/holdings'  

    #Initial API call to establish current positions for ARKK
    # need to code for an error response if API call is unsuccessfsul i.e. i.e.response.status_code == 200:
    response = requests.get(holdings_url, params = {'symbol' : 'ARKK'}).json()        #print(json.dumps(response, indent=4, sort_keys=True))
    # %% [markdown]
    # **Something for us to consider -- would it be better to utilize dataframes or databases to manipulate and analyze our data?

    # %%
    # We want to create a dataframe with the relevant 'holdings' data from the json object returned above
    initial_holdings_df = pd.DataFrame(response['holdings']).dropna(axis=0)

    

    #Check to confirm we have dropped null values in our DataFrame
    #display(initial_holdings_df.isnull().sum())

    # %% [markdown]
    # **To be done for project -- we need to find a solution for null values in our holdings dataframe as it could change 
    # and we do not necessarily want to have to dig in and figure out which value is null and what belongs there... possibly
    #  create an if/then statement for null values and how to handle them i.e. alert the user of the null value and provide 
    # options for  how to handle it.  For the purposes of our MVP, we are going to drop rows with null values since the 
    # recurring null value is for a ticker with very little impact on the portfolio.  For future consideration would be a 
    # more elegant way to handle null values, but for now we will simply drop them.  NOTE:  this will cause our weights to 
    # not equal 100, thus we must rebalance the weights so we can run our calculations. 
    ## We will save above for further exploration 

    # %%
    # For our purposes we want to focus on the 'ticker','weight', and 'company' columns of the dataframe.  This will allow us to perform historical research with the Alpaca API as well as plot the weights of the portfolio stocks.
    initial_filtered_df = initial_holdings_df[['ticker', 'weight', 'company']].sort_values(by = 'weight')

    # Note that for our Monte Carlo simulations, we will need to divide the weights column by 100 since the sum of weights for the simulation needs to be 1, and the dataframe is configured for the sum to be 100.

    initial_filtered_bar = initial_filtered_df.hvplot.bar(x='ticker', y = 'weight', hover_color = 'red', rot=90, title = 'Stock tickers and their corresponding weights in the portfolio')

<<<<<<< HEAD
    # For now we are not displaying the below chart in Streamlit for a cleaner presentation, we will be displaying
    # stacked bars chart that incorporates the weights, sharpe ratios, and std
    #st.bokeh_chart(hv.render(initial_filtered_bar, backend='bokeh'))
=======
    st.subheader('Initial ARKK portfolio analysis vs. QQQ')
    
    st.bokeh_chart(hv.render(initial_filtered_bar, backend='bokeh'))


    #display(initial_filtered_df)
>>>>>>> a7be19c2d5a86f51dbd4892367d990d47ecfe002


    # %%
    #Use data from ARKK API call to get historical quotes from Alpaca
    tickers = initial_filtered_df['ticker'].astype(str).tolist()

    timeframe = '1D'
    today = pd.Timestamp.now(tz="America/New_York")
    three_years_ago = pd.Timestamp(today - pd.Timedelta(days=1095)).isoformat()
    end_date = today
    # Do we want to use 1, 2, 3 years of historical data or more?
    start_date = three_years_ago
    # Here we are retrieving the historical data for the stocks in the ARKK portfolio.  
    # We then filter the results to leave us with closing price and ticker columns with a datetime index 
    # so we can run our analyses.

    #ARKK broken up into individual stocks:
    initial_portfolio_df = apf.get_historical_dataframe(tickers, start_date, end_date, timeframe)


    #ARKK fund as a whole:
    arkk_df = apf.get_historical_dataframe('ARKK', start_date, end_date, timeframe)

    #QQQ for comparison purposes
    qqq_df = apf.get_historical_dataframe('QQQ', start_date, end_date, timeframe)

    # %% [markdown]
    # **TBD for project -- how will we handle timeframes for our historical analyses i.e. do we 
    # want a hard coded time period or allow for user input?  Also how will this affect stocks 
    # that have no data for certain periods as well as those who have a more extensive 
    # price history.  --- For now we will hard code but could take input in the future
    # %% [markdown]
    # **One thing to consider for our daily returns calculations.. it's possible we can just 
    # set up the Monte Carlo simulation and then pull the returned daily returns to use in our 
    # risk/return analyses -- YES WE CAN!
    # %% [markdown]
    

    # %%
    # In order to use the weights from the portfolio dataframe in our Monte Carlo simulations, we will 
    # need to divide them by 100.
    # Initially we use the weights we received from our API call to retrieve ARKK's holdings -- 
    # the user will be allowed to change these and we will update them for calcuations
    # #display(initial_holdings_df.weight)
    # Dividing the weights by 100
    initial_mc_weights = list(initial_holdings_df.weight / 100)
    
    # #display(initial_mc_weights)
<<<<<<< HEAD
    num_simulations = 50
    num_trading_days = 252 * 2
=======
    num_simulations = 25
    num_trading_days = 252 
>>>>>>> a7be19c2d5a86f51dbd4892367d990d47ecfe002

    # Creating initial MC Simulation DataFrames
    # For ARKK ETF stocks (before updating)
    portfolio_initial_sim_input = mcf.configure_monte_carlo(initial_portfolio_df, initial_mc_weights, 
    num_simulations, num_trading_days)

    # For QQQ ETF (for comparison purposes)
    qqq_sim_input = mcf.configure_monte_carlo(qqq_df, [1], num_simulations, num_trading_days)

    # For the ARKK ETF
    arkk_sim_input = mcf.configure_monte_carlo(arkk_df, [1], num_simulations, num_trading_days)
    #display(portfolio_initial_sim_input.portfolio_data.tail())


    # %%
    # We will isolate the daily returns column from our MC_portfolio dataframe in order to run our risk / 
    # return analyses by merging the 'daily_return' series from our dataframe 
    initial_portfolio_daily_returns = pd.DataFrame()

    # Iterate through tickers in our MC DataFrame, filter to isolate 'daily_return' data, then concat the resulting series into 
    # dataframe we can use for our analyses on the individual stocks in the portfolio
    for ticker in tickers:
        daily_returns = portfolio_initial_sim_input.portfolio_data[ticker]['daily_return']
        initial_portfolio_daily_returns = pd.concat([initial_portfolio_daily_returns, daily_returns], axis=1)
    initial_portfolio_daily_returns.columns = tickers
    
    # Daily returns for 'QQQ' ETF for comparison
    qqq_daily_returns = qqq_sim_input.portfolio_data['QQQ']['daily_return']

    # Daily Returns for 'ARKK' ETF 
    arkk_daily_returns = arkk_sim_input.portfolio_data['ARKK']['daily_return']


    # %%
    # Calculating standard deviations of the daily returns and plotting the results.
    combined_std = pd.DataFrame()
    
    # Initial ARKK portfolio stocks
    initial_portfolio_std = initial_portfolio_daily_returns.std().sort_values()
    
    # QQQ and ARKK ETF calcs for comparison
    qqq_daily_returns_std = pd.Series(qqq_daily_returns.std())
    arkk_daily_returns_std = pd.Series(arkk_daily_returns.std())
    qqq_daily_returns_std.index = ['QQQ']
    arkk_daily_returns_std.index = ['ARKK']
    
    #Combining the std calcs into one dataframe
    combined_std = pd.concat([initial_portfolio_std, arkk_daily_returns_std, qqq_daily_returns_std])
    
    # Creating the std barplot to include in our stacked bar plot later in the program
    comparison_std_barplot = combined_std.hvplot.bar(title = 'Initial comparisons of standard deviations of daily return data (higher # = more volatile)', color = 'red',
    xlabel = 'Ticker', ylabel = 'Standard Deviation of Daily Returns', rot = 90, fontsize = {'title' : '10pt'})
    #st.bokeh_chart(hv.render(comparison_std_barplot, backend='bokeh'))

    



    # %%
    # Calculating the Sharpe Ratios for our initial portfolio stocks, ARKK and QQQ
    # First we calculate annual return data for our datasets
    year_trading_days = 252
    initial_portfolio_annual_return = initial_portfolio_daily_returns.mean() * year_trading_days
    arkk_annual_return = arkk_daily_returns.mean() * year_trading_days
    qqq_annual_return = qqq_daily_returns.mean() * year_trading_days


    # Now we calculate the annualized standard deviation
    initial_portfolio_annual_std = initial_portfolio_std * np.sqrt(year_trading_days)
    qqq_annual_std = qqq_daily_returns_std * np.sqrt(year_trading_days)
    arkk_annual_std = arkk_daily_returns_std * np.sqrt(year_trading_days)


    # Lastly, we calculate and plot the Sharpe ratios
    # Calculate the ratios
    initial_portfolio_sharpe = initial_portfolio_annual_return / initial_portfolio_annual_std
    initial_portfolio_sharpe.sort_values(inplace=True)
    qqq_sharpe = qqq_annual_return / qqq_annual_std
    arkk_sharpe = arkk_annual_return / arkk_annual_std
    combined_sharpe = pd.concat([initial_portfolio_sharpe, qqq_sharpe, arkk_sharpe]).sort_values()

    # Plot the ratios
    initial_sharpe_plot = initial_portfolio_sharpe.hvplot.bar(xlabel = 'Tickers', ylabel = 'Sharpe Ratios', title = 'Sharpe Ratios of initial portfolio stocks vs. QQQ and ARKK', color = 'red', label = 'Initial Portfolio Stocks', rot = 90)
    arkk_sharpe_plot = arkk_sharpe.hvplot.bar(xlabel = 'Tickers', ylabel = 'Sharpe Ratios', title = 'Sharpe Ratio of ARKK', label = 'ARKK', color='blue', rot = 90)
    qqq_sharpe_plot = qqq_sharpe.hvplot.bar(xlabel = 'Tickers', ylabel = 'Sharpe Ratios', title = 'Sharpe Ratio of QQQ', label = 'QQQ', color='purple', rot=90)
    
    # Combined plot for the overlay
    combined_sharpe_plot = initial_sharpe_plot * arkk_sharpe_plot * qqq_sharpe_plot
    #st.bokeh_chart(hv.render(combined_sharpe_plot, backend='bokeh'))

  




    # %%
    # Creating a stacked bar chart for visualization purposes -- values for 'std' and 'sharpe' are multiplied to normalize the values
    #Concatenate the dataframes and format the resulting df
    # Assigining the index of our filtered df for a smooth concat
    initial_filtered_df.index = initial_filtered_df.ticker
    
    combined_plot_df = pd.concat([combined_sharpe, combined_std, initial_filtered_df], axis = 1)
    combined_plot_df.rename(columns = {0 : 'sharpe', 1 : 'std'}, inplace=True)
    combined_plot_df.sort_values(by='std', inplace=True)
    
    # We multiply the std and sharpe values in order to normalize the data and create a useful visual,
    # with the understanding these are not the actual sharpe and std values in our plot
    combined_plot_df['sharpe'] = combined_plot_df['sharpe'] * 3
    combined_plot_df['std'] = combined_plot_df['std'] * 75
    stacked_bars_plot = combined_plot_df.hvplot.bar(rot=90, y=['std', 'sharpe', 'weight'], legend = True, stacked=True, title = 'Sharpe Ratios and Standard Deviations for our portfolio stocks')
    
    # Plot the chart in Streamlit
    st.bokeh_chart(hv.render(stacked_bars_plot, backend='bokeh'))

    # %%
    # Calculating and plotting historical cumulative returns for the stocks in ARKK
    initial_portfolio_cumprod = (1 + initial_portfolio_daily_returns).cumprod()
    initial_portfolio_cum_plot = initial_portfolio_cumprod.hvplot(kind = 'line', rot=90, title = 'Cumulative returns for the individual stocks in ARKK over time', 
                                                                ylabel = 'Returns', xlabel = 'Date', legend = 'left', fontsize = {'legend' : '8pt'}, frame_height = 250, frame_width = 750)
    # For 'ARKK' as  whole
    arkk_returns_cumprod = (1 + arkk_daily_returns).cumprod()
    arkk_cum_plot = arkk_returns_cumprod.hvplot(kind = 'line', rot=90, title = 'Historical cumulative returns for ARKK vs QQQ', ylabel = 'Returns', xlabel = 'Date', label = 'ARKK', color='blue')
    # For 'QQQ' as a whole
    qqq_returns_cumprod = (1 + qqq_daily_returns).cumprod()
    qqq_cum_plot = qqq_returns_cumprod.hvplot(kind = 'line', rot=90, title = 'Cumulative returns for QQQ', ylabel = 'Returns', xlabel = 'Date', label = 'QQQ')

    # Comparing the historcial returns of QQQ vs. ARKK for comparison -- TBD if we want to display these charts
    # in our main program
    qqq_arkk_cumplot = arkk_cum_plot * qqq_cum_plot
    #st.bokeh_chart(hv.render(initial_portfolio_cum_plot, backend='bokeh'))
    #st.bokeh_chart(hv.render(qqq_arkk_cumplot, backend='bokeh'))


    # %% [markdown]
    # ### Calling the Monte Carlo functions and running projections -- 

    # %%
    # Run the simulations:
    # Initially ARKK simulation data will consist of the initial holdings of the ETF -- 
    # future simulations will be variable depending on how the user wants to manipulate the portfolio)
    
    arkk_initial_sim_returns = mcf.run_monte_carlo(portfolio_initial_sim_input)
    qqq_sim_returns = mcf.run_monte_carlo(qqq_sim_input)

    
    # %%
    initial_return_totals = pd.DataFrame(arkk_initial_sim_returns.iloc[-1, :])
    #display(initial_return_totals)

    # %%
    # Plotting the median projected returns via the MCForecast projections
    qqq_median_outcomes = qqq_sim_returns.median(axis=1)
    arkk_intial_median_outcomes = arkk_initial_sim_returns.median(axis=1)
    qqq_median_plot = qqq_median_outcomes.hvplot(ylabel = 'Median returns', xlabel = 'Days of projection', title = f'Median returns from {num_simulations} simulations over {num_trading_days} trading days for QQQ vs ARKK', label = 'QQQ', color='red')
    arkk_median_initial_plot = arkk_intial_median_outcomes.hvplot(ylabel = 'Median returns', xlabel = 'Days of projection', title = f'Median returns from {num_simulations} simulations over {num_trading_days} trading days for QQQ vs ARKK', label = 'ARKK', color='blue')
    initial_combined_median_plot = qqq_median_plot * arkk_median_initial_plot
    st.bokeh_chart(hv.render(initial_combined_median_plot, backend='bokeh'))


    # %%
    # Plotting distribrution and confidence intervals from Monte Carlo Simulation
    # This is the plot for the simulations using the individual stocks within ARKK and can be manipulated...
    # this plot will be variable whereas the 'ARKK' and 'QQQ' PLOT
    portfolio_intial_distribution_plot = mcf.plot_distribution(portfolio_initial_sim_input)
<<<<<<< HEAD
    st.subheader('ARKK simulation results distribution plot')
    st.plotly_chart(portfolio_intial_distribution_plot)
=======
    st.subheader('Distribution plot of simulated returns for ARKK')
    st.plotly_chart(portfolio_intial_distribution_plot, sharing="streamlit")
>>>>>>> a7be19c2d5a86f51dbd4892367d990d47ecfe002



    # %%
    qqq_distribution_plot = mcf.plot_distribution(qqq_sim_input)
<<<<<<< HEAD
    st.subheader('QQQ simulation results distribution plot')
    st.plotly_chart(qqq_distribution_plot)
=======
    st.subheader('Distribution plot of simulated returns for QQQ')
    st.plotly_chart(qqq_distribution_plot, sharing="streamlit", title = 'Distribution of cumulative returns across all simulations for QQQ')

>>>>>>> a7be19c2d5a86f51dbd4892367d990d47ecfe002

    
    # %%
    # Describe the MCForecast Summary
    portfolio_initial_simulation_summary = mcf.get_monte_summary(portfolio_initial_sim_input)
    qqq_simulation_summary = mcf.get_monte_summary(qqq_sim_input)
    
<<<<<<< HEAD
    # Display tables with the summary data
    st.subheader('Initial portfolio returns summary')
    st.table(portfolio_initial_simulation_summary)
    
    st.subheader('QQQ summary simulation results')
=======
    st.subheader('Summary table of Cumulative Returns from our simulations for ARKK')
    st.table(portfolio_initial_simulation_summary)
    st.subheader('Summary table of Cumulative Returns from our simulations for QQQ')
>>>>>>> a7be19c2d5a86f51dbd4892367d990d47ecfe002
    st.table(qqq_simulation_summary)

    # Adding in inputs to alter the portfolio for new analyses
    st.subheader('Please select below if you would like to drop or change any stocks to see how it would affect the portfolio:')
    
    with st.form('drop_stock'):
        updated_holdings_df = pd.DataFrame()
        drop_choice = st.selectbox('Select the stock to drop', options=initial_holdings_df['ticker'])
        updated_holdings_df = initial_holdings_df[initial_holdings_df.ticker != drop_choice]
        updated_holdings_df.weight = updated_holdings_df.weight + ((100 - updated_holdings_df.weight.sum()) / len(updated_holdings_df.index))
        update_analyses = st.form_submit_button('Run analyses on updated portfolio')
        if update_analyses:
            runUpdatedAnalysis(updated_holdings_df, initial_filtered_bar, comparison_std_barplot, combined_sharpe_plot, stacked_bars_plot, initial_portfolio_cum_plot, initial_combined_median_plot, portfolio_intial_distribution_plot)
    
    with st.form('switch_stock'):
        
        
        change_choices = []
        updated_holdings_df = pd.DataFrame()
<<<<<<< HEAD
        change_choices = list(st.multiselect('Select the stock(s) to change out', options = initial_holdings_df['ticker']))
        with st.container():
            new_stocks = list(st.text_input('Stocks to replace with', 'e.g. SPY,TLT,QQQ'))
        updated_holdings_df = initial_holdings_df
        i = 0
        while i < len(change_choices):
            updated_holdings_df['ticker'].replace({change_choices[i] : new_stocks[i]}, inplace=True)
            i += 1
            
        update_analyses = st.form_submit_button('Run analyses on updated portfolio')
        if update_analyses:
=======
        updated_holdings_df = initial_holdings_df
        change_choices = st.multiselect('Select up to 3 stocks to change out', options = initial_holdings_df['ticker'])
        col1, col2, col3 = st.columns(3)
        col1 = st.text_input('Replacement stock 1')
        col2 = st.text_input('Replacement stock 2 (if applicable)')
        col3 = st.text_input('Replacement stock 3 (if applicable)')
        




        update_analyses = st.form_submit_button('Run analyses on updated portfolio')
        if update_analyses:
            updated_holdings_df['ticker'].replace({change_choices[0] : col1}, inplace=True)
            if change_choices[1]:
                updated_holdings_df['ticker'].replace({change_choices[1] : col2}, inplace=True)
            if change_choices[2]:
                updated_holdings_df['ticker'].replace({change_choices[2] : col3}, inplace=True)

            if len(change_choices) > 3:
                st.write('No more than 3 stocks can be replaced')
                st.stop()
            #st.write('Once a stock is dropped the change out stocks function will not work')
>>>>>>> a7be19c2d5a86f51dbd4892367d990d47ecfe002
            runUpdatedAnalysis(updated_holdings_df, initial_filtered_bar, comparison_std_barplot, combined_sharpe_plot, stacked_bars_plot, initial_portfolio_cum_plot, initial_combined_median_plot, portfolio_intial_distribution_plot)    
            #st.stop()
    
    #return initial_holdings_df, initial_filtered_bar, comparison_std_barplot, combined_sharpe_plot, stacked_bars_plot, initial_portfolio_cum_plot, initial_combined_median_plot, portfolio_intial_distribution_plot




runFirstAnalysis()



