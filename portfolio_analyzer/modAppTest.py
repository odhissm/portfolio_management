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
from utils import MonteCarloFunctions as mcf
from utils import AlpacaFunctions as apf
import datetime as dt
import hvplot
import hvplot.pandas
import matplotlib.pyplot as plt
import plotly as pty
import plotly.express as px
import bokeh
import streamlit as st
from utils.initial_analyses import runFirstAnalysis
from utils.updated_analysis import runUpdatedAnalysis

# %% [markdown]
# **Below we are going to define functions we will be using repeatedly within the project -- will probably need to be moved to a separate .py file in order to "modularize" our app.  We will then be able to import the relevant functions from the separate file.    # Def function that will be the loop of the app.  Takes a newPortfolioData parameter that will run our analyses on the new portfolio data that results from user input


runFirstAnalysis()
runUpdatedAnalysis()



