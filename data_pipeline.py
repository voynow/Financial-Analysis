# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 11:45:07 2020

@author: voyno
"""

import pandas as pd

# import preprocessing, fill_nan scripts
from preprocessing import preprocess
from fill_nans import fill_nans


def run_pipeline(files):

    # convert file to list
    if type(files) is str:
        files = [files]
    
    # Read raw data from yfinance api for first file in list
    dfs =[pd.read_csv(file) for file in files]
    
    # preprocessing on raw data
    df = preprocess(dfs)

    # removing nan values
    df = fill_nans(df)

    return df
