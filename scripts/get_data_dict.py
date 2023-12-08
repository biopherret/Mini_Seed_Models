from sqlalchemy.engine import URL
from sqlalchemy import create_engine
import sqlalchemy as sa
import pyodbc
import pandas as pd

import sys
sys.path.append('.../.../')
from functions import model_general as mg

conn_str = (
    r'driver={SQL Server};'
    r'server=LEXI_DESKTOP\SQLEXPRESS;' #server name
    r'database=FygensonLabData;' #database name
    r'trusted_connection=yes;'
    )

#using SQLAlchemy to avoid a UserWarning
connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": conn_str})
engine = create_engine(connection_url) #create SQLAlchemy engine object

cnxn = pyodbc.connect(conn_str) #connect to server using pyodbc
cursor = cnxn.cursor()

def run_quary(quary_str):
    '''Run a quary and return the output as a pandas datafrme

    Args:
        quary_str (str): quary string (not case sensitive, SQL strings need to be enclosed in single quotes)

    Returns:
        dataframe: quary output
    '''
    with engine.begin() as conn:
        return pd.read_sql_query(sa.text(quary_str), conn)
    
if __name__ == '__main__':
    
    data = {'os p3024': {2023042700: {}, 2023042701: {}, 2023042702: {}, 2023042800: {}, 2023042801: {}, 2023042802: {}},
            'os s768': {2022102804: {}, 2022110107: {},  2022110108: {}},
            'os s576': {2022102803: {}, 2022110105: {}, 2022110106: {}},
            'os s384': {2022102802: {}, 2022110103: {}, 2022110104: {}},
            'ts p3024': {2023042703: {}, 2023042704: {}, 2023042705: {}},
            'ts s768': {2022111404: {}, 2022111405: {}, 2022111503: {}},
            'ts s576': {2023041900: {}, 2023041901: {}, 2023041902: {}},
            'ts s384': {2023042600: {}, 2023042601: {}, 2023042602: {}}}

    seed_len_types = [ 'p3024', 's768', 's576', 's384']
    pixel_per_um = 15.4792

    #get sample data
    for seed_len in seed_len_types:
        #one sided seed
        for slide_sample_id in data[f'os {seed_len}']:
            data_df = run_quary(f'Select * From length_distributions Where slide_sample_id = {slide_sample_id};').set_index('slide_sample_id') #get all one sided lengths 
            data[f'os {seed_len}'][slide_sample_id]['length_distribution'] = [len / pixel_per_um for len in data_df.loc[:]['lengths'].tolist()]

        #two sided seed
        for slide_sample_id in data[f'ts {seed_len}']:
            re_data_df = run_quary(f"Select * From length_distributions Where slide_sample_id = {slide_sample_id} and is_in_ts = 1.0 and length_type = 're';").set_index('slide_sample_id') #only select the re lengths which are part of a ts tube
            data[f'ts {seed_len}'][slide_sample_id]['re_length_distribution'] = [len / pixel_per_um for len in re_data_df.loc[:]['lengths'].tolist()]
            se_data_df = run_quary(f"Select * From length_distributions Where slide_sample_id = {slide_sample_id} and is_in_ts = 1.0 and length_type = 'se';").set_index('slide_sample_id') #only select the re lengths which are part of a ts tube
            data[f'ts {seed_len}'][slide_sample_id]['se_length_distribution'] = [len / pixel_per_um for len in se_data_df.loc[:]['lengths'].tolist()]

    #save data as dict
    mg.write_json(data, 'data_dict.json')