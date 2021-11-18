# backend code for portfolio analyzer
# business logic of the application
# access to the database for CRUD operations
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
import hvplot.pandas
import panel as pn
from utils import MonteCarloFunctions as mcf
from utils import AlpacaFunctions as apf
import datetime as dt
import hvplot
import hvplot.pandas
import matplotlib.pyplot as plt
import plotly as pty
import plotly.express as px
import bokeh



#    
def pull_funds_portfolio(holdings_symbol, holdings_url):
    #return db_bkend.import_portfolio_holding(holdings_symbol, holdings_url)
    
    if (holdings_symbol == 'ARKK'):
        holdings_url = 'https://arkfunds.io/api/v2/etf/holdings' 
        
    elif (holdings_symbol == 'YOLO'):
         holdings_url = ''
    
    elif (holdings_symbol == 'MSOS'):
        holdings_url = ''
    
    elif (holdings_symbol == 'IVY'):
        holdings_url = ''
        
    else:
        holdings_url = ''
        
        
    funds_portfolio = requests.get(holdings_url, params = {'symbol' : holdings_symbol}).json()
    
    return funds_portfolio



