import sqlite3
import fire
import questionary
import sqlalchemy
import pandas as pd
from pathlib import Path

import os
import requests
import json
import alpaca_trade_api as tradeapi
from sqlite3 import OperationalError, IntegrityError, ProgrammingError
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from model import mvc_exceptions as mvc_exc


db_name = 'mydb'

# Create a temporary sqlite database

def connect_to_db(db=None):
    """
    Connect to sqlite db. Create the database if there isn't one yet.
    
    Parameters: 
                db : str
                    database name. If none is present then create an inmemory db
                    
    Returns:
            connection: sqlite.Connection
                connection 
    """
    
    if db is None:
        database_connection_string = 'sqlite:///'
        con3 = sqlite3.connect(":memory:")
        print('This is a new connection to in memory SQLite db...')
    else:
        database_connection_string = 'sqlite:///{}.db'.format(db)
        con3 = sqlite3.connect('sqlite:///{}.db'.format(db))
        print('This is a new connection to SQLite db...')
    
    # Create an engine/connection to interact with the database
    connection = sqlalchemy.create_engine(database_connection_string)
    #connection = con3
    return connection


    
# Use this function to wrap commit in a try block  
def connect(func):
    """
    Used to open or reopen database connection when needed
    
    parameters: 
                func: function which performs the db query
                
    returns:
                inner func: function
    """
    def inner_func(conn, *args, **kwargs):
        try:
            # quick query to try
            conn.execute(
                'SELECT name FROM sqlite_temp_master WHERE type="table";'
            )
        except (AttributeError, ProgrammingError):
            conn = connect_to_db(db_name)
        return func(conn, *args, **kwargs)
    return inner_func


# Use this function to make explicit disconnection from the database
def disconnect_from_db(db=None, conn=None):
    if db is not db_name:
        print("You are trying to disconnect from a wrong db")
    if conn is not None:
        conn.close()
        
        
# use this function to prevent SQL injection
def scrub(input_string):
    """
    Clean input string (to prevent SQL injection)
    
    parameters:
                input_string : str
                
    returns:
                str
    """
    
    return ''.join(k for k in input_string if k.isalnum())


"""CREATE table from DataFrame

The CREATE operation creates a new table in the database using the given DataFrame.
The table is replaced by the new data if it already exists.
"""

@connect
def create_table_from_dataframe(conn, table_name, table_data_df):
    table_name = scrub(table_name)
    str = '{}'.format(table_name)
    try:
        table_data_df.to_sql(str, con=conn, index=True, if_exists='replace')        
        
    except OperationalError as e:
        print(e)
    
"""READ database table into DataFrame

The READ operation will read the entire table from the database into a new DataFrame.
Then it will print the DataFrame.
"""
@connect
def read_table_into_dataframe(conn, table_name):
    table_name = scrub(table_name)
    str = '{}'.format(table_name)
    
    try:
        results_dataframe = pd.read_sql_table(str, con=conn)
        print(f"{table_name} Data:")
        print(results_dataframe)
    except OperationalError as e:
        print(e)     
        
        
    
"""CREATE table from SQL

This CREATE operation creates a new table in the database using the given SQL statement.
The table is replaced by the new data if it already exists.
"""    

        
@connect
def create_table(conn, table_name):
    table_name = scrub(table_name)
    sql = 'CREATE TABLE {} (rowid INTEGER PRIMARY KEY AUTOINCREMENT,'\
    'ticker TEXT UNIQUE, weight REAL, company TEXT)'.format(table_name)
    try:
        conn.execute(sql)
    except OperationalError as e:
        print(e)
        

"""function to INSERT data into table using sql

"""        
@connect
def insert_one(conn, ticker, weight, company, table_name):
    table_name = scrub(table_name)
    sql = "INSERT INTO {} ('ticker', 'weight', 'company') VALUES (?, ?, ?)"\
        .format(table_name)
    try:
        conn.execute(sql, ( ticker, weight, company))
        #conn.commit()
    except IntegrityError as e:
        raise mvc_exc.AssetAlreadyStored(
            '{}: "{}" already stored in table "{}"'.format(e, ticker, table_name))

        
"""function to INSERT many records into table using sql

"""                
@connect
def insert_many(conn, assets, table_name):
    table_name = scrub(table_name)
    sql = "INSERT INTO {} ('ticker', 'weight', 'company') VALUES (?, ?, ?)"\
        .format(table_name)
    entries = list()
    for x in assets:
        entries.append((x['ticker'], x['weight'], x['company']))
    try:
        conn.executemany(sql, entries)
        conn.commit()
    except IntegrityError as e:
        print('{}: at least one in {} was already stored in table "{}"'
              .format(e, [x['ticker'] for x in assets], table_name))
        

def tuple_to_dict(mytuple):
    mydict = dict()
    mydict['id'] = mytuple[0]
    mydict['ticker'] = mytuple[1]
    mydict['weight'] = mytuple[2]
    mydict['company'] = mytuple[3]
    return mydict


"""function to READ table using sql

"""

@connect
def select_one(conn, asset_ticker, table_name):
    table_name = scrub(table_name)
    asset_ticker = scrub(asset_ticker)
    sql = 'SELECT * FROM {} WHERE ticker="{}"'.format(table_name, asset_ticker)
    c = conn.execute(sql)
    result = c.fetchone()
    if result is not None:
        return tuple_to_dict(result)
    else:
        raise mvc_exc.AssetNotStored(
            'Can\'t read "{}" because it\'s not stored in table "{}"'
            .format(asset_ticker, table_name))

        
"""function to SELECT all records from the table using SQL

"""
        
@connect
def select_all(conn, table_name):
    table_name = scrub(table_name)
    sql = 'SELECT * FROM {}'.format(table_name)
    c = conn.execute(sql)
    results = c.fetchall()
    return list(map(lambda x: tuple_to_dict(x), results))        
        

"""function to UPDATE a record in the table using sql

"""    
@connect
def update_one(conn, ticker, weight, company, table_name):
    table_name = scrub(table_name)
    sql_check = 'SELECT EXISTS(SELECT 1 FROM {} WHERE ticker=? LIMIT 1)'\
        .format(table_name)
    sql_update = 'UPDATE {} SET weight=?, company=? WHERE ticker=?'\
        .format(table_name)
    c = conn.execute(sql_check, (ticker,))  # we need the comma
    result = c.fetchone()
    if result[0]:
        conn.execute(sql_update, (weight, company, ticker))
        #conn.commit()
    else:
        raise mvc_exc.AssetNotStored(
            'Can\'t update "{}" because it\'s not stored in table "{}"'
            .format(ticker, table_name))    

        
"""DELETE a record in the table using sql

"""        
        
        
@connect
def delete_one(conn, ticker, table_name):
    table_name = scrub(table_name)
    sql_check = 'SELECT EXISTS(SELECT 1 FROM {} WHERE ticker=? LIMIT 1)'\
        .format(table_name)
    table_name = scrub(table_name)
    sql_delete = 'DELETE FROM {} WHERE ticker=?'.format(table_name)
    c = conn.execute(sql_check, (ticker,))  # we need the comma
    result = c.fetchone()
    if result[0]:
        conn.execute(sql_delete, (ticker,))  # we need the comma
        #conn.commit()
    else:
        raise mvc_exc.AssetNotStored(
            'Can\'t delete "{}" because it\'s not stored in table "{}"'
            .format(ticker, table_name))        
        
    
def import_portfolio_holding(holdings_symbol, holdings_url):
        
    # make an API call to get the jason file of the holding and store in a variable 'response'
    response = requests.get(holdings_url, params = {'symbol' : holdings_symbol}).json()
    
    # Create a dataframe of the holding response
    holdings_df = pd.DataFrame(response['holdings'])
    
    #create a new dataframe from holdings_df and select only the following columns; ticker, weight and company    
    holdings_ticker_weight_df = holdings_df[['ticker', 'weight', 'company']]
    
    return holdings_ticker_weight_df
    
