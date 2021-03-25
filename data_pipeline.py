# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 11:45:07 2020

@author: voyno
"""

import pandas as pd

# import preprocessing, fill_nan scripts
from preprocessing import preprocess
from fill_nans import fill_nans


def run_pipeline(file):
    if type(file) is str: # One file
        # Read raw data from yfinance api
        df = pd.read_csv(file)
    
        # preprocessing on raw data
        df = preprocess(df)
    
        # removing nan values
        df = fill_nans(df)
    else: # Multiple files
        # Read raw data from yfinance api for first file in list
        df = pd.read_csv(file[0])
        
        # preprocessing on raw data
        df = preprocess(df)
        
        # removing nan values
        df = fill_nans(df)
        
        # looping over the rest of the files in the list
        for i in range(1, len(file)):
            # Read raw data
            next_df = pd.read_csv(file[i])
            # preprocess
            next_df = preprocess(next_df)
            # fill nans
            next_df = fill_nans(next_df)
            # concatenate to df
            # inner join: only shared columns are kept
            df = pd.concat([df, next_df], join="inner")   

    # now data is prepared to model
    return df
