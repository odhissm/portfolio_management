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
import datetime as dt
import holoviews as hv
import panel as pn
import utils.MonteCarloFunctions as mcf
import utils.AlpacaFunctions as apf
from utils.updated_analysis import runUpdatedAnalysis
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
    # need to code for an error response if API call is unsuccessfsul i.e. i.e.response.status_code == 200: -- for future iterations
    response = requests.get(holdings_url, params = {'symbol' : 'ARKK'}).json()        #print(json.dumps(response, indent=4, sort_keys=True))
    # %% [markdown]
    # **Something for us to consider -- would it be better to utilize dataframes or databases to manipulate and analyze our data? -- we have a baseline coded for database interaction if we decide to do so in the future

    # %%
    # We want to create a dataframe with the relevant 'holdings' data from the json object returned above
    initial_holdings_df = pd.DataFrame(response['holdings']).dropna(axis=0)

    # %% [markdown]
    # **To be done for project -- we need to find a solution for null values in our holdings dataframe as it could change and we do not necessarily want to have to dig in and figure out which value is null and what belongs there... possibly create an if/then statement for null values and how to handle them i.e. alert the user of the null value and provide options for  how to handle it.  For the purposes of our MVP, we are going to drop rows with null values since the recurring null value is for a ticker with very little impact on the portfolio.  For future consideration would be a more elegant way to handle null values, but for now we will simply drop them.  NOTE:  this will cause our weights to not equal 100, thus we must rebalance the weights so we can run our calculations.

    # %%
    # For our purposes we want to focus on the 'ticker','weight', and 'company' columns of the dataframe.  This will allow us to perform historical research with the Alpaca API as well as plot the weights of the portfolio stocks.
    initial_filtered_df = initial_holdings_df[['ticker', 'weight', 'company']].sort_values(by = 'weight')

    # Note that for our Monte Carlo simulations, we will need to divide the weights column by 100 since the sum of weights for the simulation needs to be 1, and the dataframe is configured for the sum to be 100.

    initial_filtered_bar = initial_filtered_df.hvplot.bar(x='ticker', y = 'weight', hover_color = 'red', rot=90, title = 'Stock tickers and their corresponding weights in the portfolio')

    st.subheader('Initial ARKK portfolio analysis vs. QQQ')
    
    st.bokeh_chart(hv.render(initial_filtered_bar, backend='bokeh'))



    # %%
    #Use data from ARKK API call to get historical quotes from Alpaca
    tickers = initial_filtered_df['ticker'].astype(str).tolist()

    timeframe = '1D'
    today = pd.Timestamp.now(tz="America/New_York")
    three_years_ago = pd.Timestamp(today - pd.Timedelta(days=1095)).isoformat()
    end_date = today
    # Do we want to use 1, 2, 3 years of historical data or more? -- for now we will use 3 years
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

   
    # %%
    # In order to use the weights from the portfolio dataframe in our Monte Carlo simulations, we will need to divide them by 100.
    # Initially we use the weights we received from our API call to retrieve ARKK's holdings -- 
    # the user will be allowed to change these and we will update them for calcuations
    # #display(initial_holdings_df.weight)
    # Dividing the weights by 100
    initial_mc_weights = list(initial_holdings_df.weight / 100)
    
    # #display(initial_mc_weights)
    num_simulations = 25
    num_trading_days = 252 

    # Creating initial MC Simulation DataFrames
    # For ARKK ETF stocks (before updating)
    portfolio_initial_sim_input = mcf.configure_monte_carlo(initial_portfolio_df, initial_mc_weights, num_simulations, num_trading_days)

    # For QQQ ETF (for comparison purposes)
    qqq_sim_input = mcf.configure_monte_carlo(qqq_df, [1], num_simulations, num_trading_days)

    # For the ARKK ETF
    arkk_sim_input = mcf.configure_monte_carlo(arkk_df, [1], num_simulations, num_trading_days)
    #display(portfolio_initial_sim_input.portfolio_data.tail())


    # %%
    # We will isolate the daily returns column from our MC_portfolio dataframe in order to run our risk / return analyses by
    # merging the 'daily_return' series from our dataframe 
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
    initial_portfolio_std = initial_portfolio_daily_returns.std().sort_values()
    qqq_daily_returns_std = pd.Series(qqq_daily_returns.std())
    arkk_daily_returns_std = pd.Series(arkk_daily_returns.std())
    qqq_daily_returns_std.index = ['QQQ']
    arkk_daily_returns_std.index = ['ARKK']
    combined_std = pd.concat([initial_portfolio_std, arkk_daily_returns_std, qqq_daily_returns_std])
    comparison_std_barplot = combined_std.hvplot.bar(title = 'Initial comparisons of standard deviations of daily return data (higher # = more volatile)', color = 'red',
    xlabel = 'Ticker', ylabel = 'Standard Deviation of Daily Returns', rot = 90, fontsize = {'title' : '10pt'})
    st.bokeh_chart(hv.render(comparison_std_barplot, backend='bokeh'))

    



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
    combined_sharpe_plot = initial_sharpe_plot * arkk_sharpe_plot * qqq_sharpe_plot
    st.bokeh_chart(hv.render(combined_sharpe_plot, backend='bokeh'))

  




    # %%
    # Creating a stacked bar chart for visualization purposes -- values for 'std' and 'sharpe' are multiplied to normalize the values
    #Concatenate the dataframes and format the resulting df
    initial_filtered_df.index = initial_filtered_df.ticker
    combined_plot_df = pd.concat([combined_sharpe, combined_std, initial_filtered_df], axis = 1)
    combined_plot_df.rename(columns = {0 : 'sharpe', 1 : 'std'}, inplace=True)
    combined_plot_df.sort_values(by='std', inplace=True)
    combined_plot_df['sharpe'] = combined_plot_df['sharpe'] * 3
    combined_plot_df['std'] = combined_plot_df['std'] * 75
    stacked_bars_plot = combined_plot_df.hvplot.bar(rot=90, y=['std', 'sharpe', 'weight'], legend = True, stacked=True, title = 'Sharpe Ratios and Standard Deviations for our portfolio stocks')
    st.bokeh_chart(hv.render(stacked_bars_plot, backend='bokeh'))


    # %% [markdown]
    # **At some point would we like to also plot returns for QQQ or S&P and ARKK and some other fund for reference?

    # %%
    # Calculating and plotting cumulative returns for the stocks in ARKK
    initial_portfolio_cumprod = (1 + initial_portfolio_daily_returns).cumprod()
    initial_portfolio_cum_plot = initial_portfolio_cumprod.hvplot(kind = 'line', rot=90, title = 'Cumulative returns for the individual stocks in ARKK over time', 
                                                                ylabel = 'Returns', xlabel = 'Date', legend = 'left', fontsize = {'legend' : '8pt'}, frame_height = 250, frame_width = 750)
    # For 'ARKK' as  whole
    arkk_returns_cumprod = (1 + arkk_daily_returns).cumprod()
    arkk_cum_plot = arkk_returns_cumprod.hvplot(kind = 'line', rot=90, title = 'Historical cumulative returns for ARKK vs QQQ', ylabel = 'Returns', xlabel = 'Date', label = 'ARKK', color='blue')
    # For 'QQQ' as a whole
    qqq_returns_cumprod = (1 + qqq_daily_returns).cumprod()
    qqq_cum_plot = qqq_returns_cumprod.hvplot(kind = 'line', rot=90, title = 'Cumulative returns for QQQ', ylabel = 'Returns', xlabel = 'Date', label = 'QQQ')

    qqq_arkk_cumplot = arkk_cum_plot * qqq_cum_plot
    st.bokeh_chart(hv.render(initial_portfolio_cum_plot, backend='bokeh'))
    st.bokeh_chart(hv.render(qqq_arkk_cumplot, backend='bokeh'))


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
    st.subheader('Distribution plot of simulated returns for ARKK')
    st.plotly_chart(portfolio_intial_distribution_plot, sharing="streamlit")



    # %%
    qqq_distribution_plot = mcf.plot_distribution(qqq_sim_input)
    st.subheader('Distribution plot of simulated returns for QQQ')
    st.plotly_chart(qqq_distribution_plot, sharing="streamlit", title = 'Distribution of cumulative returns across all simulations for QQQ')


    
    # %%
    # Describe the MCForecast Summary
    portfolio_initial_simulation_summary = mcf.get_monte_summary(portfolio_initial_sim_input)
    qqq_simulation_summary = mcf.get_monte_summary(qqq_sim_input)
    
    st.subheader('Summary table of Cumulative Returns from our simulations for ARKK')
    st.table(portfolio_initial_simulation_summary)
    st.subheader('Summary table of Cumulative Returns from our simulations for QQQ')
    st.table(qqq_simulation_summary)

    # Adding in inputs to alter the portfolio for new analyses
    st.subheader('Please select below if you would like to drop or change any stocks to see how it would affect the portfolio:')
    
    # Here we are creating a 'form' within Streamlit to isolate the drop_stock functionality
    with st.form('drop_stock'):
        updated_holdings_df = pd.DataFrame()
        # Allowing the user to choose which stock to drop
        drop_choice = st.selectbox('Select the stock to drop', options=initial_holdings_df['ticker'])
        # Updating the portfolio and weights
        updated_holdings_df = initial_holdings_df[initial_holdings_df.ticker != drop_choice]
        updated_holdings_df.weight = updated_holdings_df.weight + ((100 - updated_holdings_df.weight.sum()) / len(updated_holdings_df.index))
        update_analyses = st.form_submit_button('Run analyses on updated portfolio')
        # If run analyses button is clicked, run the following:
        if update_analyses:
            runUpdatedAnalysis(updated_holdings_df, initial_filtered_bar, comparison_std_barplot, combined_sharpe_plot, stacked_bars_plot, initial_portfolio_cum_plot, initial_combined_median_plot, portfolio_intial_distribution_plot)
            updated_holdings_df = initial_holdings_df
    
    # Here we are creating a 'form' within Streamlit to isolate the drop_stock functionality
    with st.form('switch_stock', clear_on_submit=True):
        # Creating the framework for allowing up to 3 stocks to be switched out from our intial portfolio
        change_choices = []
        updated_holdings_df = pd.DataFrame()
        updated_holdings_df = initial_holdings_df
        change_choices = st.multiselect('Select up to 3 stocks to change out', options = initial_holdings_df['ticker'])
        col1, col2, col3 = st.columns(3)
        col1 = st.text_input('Replacement stock 1')
        col2 = st.text_input('Replacement stock 2 (if applicable)')
        col3 = st.text_input('Replacement stock 3 (if applicable)')
        
        # If analyze portfolio button is clicked, switch out the stocks, re-run the simulations, and then clear the form
        update_analyses = st.form_submit_button('Run analyses on updated portfolio')
        if update_analyses:
            updated_holdings_df['ticker'].replace({change_choices[0] : col1}, inplace=True)
            if col2 != '':
                updated_holdings_df['ticker'].replace({change_choices[1] : col2}, inplace=True)
            if col3 != '':
                updated_holdings_df['ticker'].replace({change_choices[2] : col3}, inplace=True)

            # If the user tries to select more than 3 stocks to replace, throw an error and stop the program.
            if len(change_choices) > 3:
                st.write('No more than 3 stocks can be replaced')
                st.stop()            
            
            runUpdatedAnalysis(updated_holdings_df, initial_filtered_bar, comparison_std_barplot, combined_sharpe_plot, stacked_bars_plot, initial_portfolio_cum_plot, initial_combined_median_plot, portfolio_intial_distribution_plot)    
            updated_holdings_df = initial_holdings_df
        
# One thing we did not get to is creating a better 'main' app layout that would further modularize the app


runFirstAnalysis()



