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
from utils.initial_analyses import runFirstAnalysis 
from utils.updated_analysis import runUpdatedAnalysis
import datetime as dt
import hvplot
import hvplot.pandas
import matplotlib.pyplot as plt
import plotly as pty
import plotly.express as px
import bokeh
import streamlit as st

# We will first invoke the runFirstAnalysis and then use the returned values to runUpdatedAnalysis
def main():
    runFirstAnalysis()
    runUpdatedAnalysis(runFirstAnalysis)
    
    
main()