# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 20:55:53 2020

@author: voyno
"""

import pandas as pd
import numpy as np


def prepare_dat(df):

    """ restructures asset names within the dataframe """

    # add placeholder index to edge case column names
    for col in df.columns:
        if "." not in col:
            df.rename(columns={col: col + ".0"}, inplace=True)

    # access symbols and append to column names
    columns_symbols = df.loc[df.index[0]].values
    df.columns = [df.columns[i].split(".")[0] + "." + columns_symbols[i] for i in range(len(df.columns))]

    # remove redundant row of symbols
    df.drop([df.index[i] for i in [0, 1]], axis=0, inplace=True)

    return df.astype('float')


def remove_nan_columns(threshold, df):

    """ Remove columns with nan values occurring greater than some predetermined threshold """

    threshold_count = threshold * df.shape[0]

    print()
    print("*" * 58 + "\n\tProcessing series consisting of NAN values\n" + "*" * 58, "\n")

    # find number of nan values in each column and compare to threshold
    nan_matrix = np.array([np.sum(np.isnan(df[column].values)) for column in df.columns])
    nan_columns = np.where(nan_matrix > threshold_count)[0]
    nan_column_names = df.columns[nan_columns]

    # access symbols from columns with nan values > threshold
    symbols_with_nan_cols = [nan_column_names[i].split(".")[1] for i in range(len(nan_column_names))]
    unique_symbols = np.unique(symbols_with_nan_cols)

    # create string of symbols to drop for user output
    stock_drop_string = ""
    for item in unique_symbols:
        stock_drop_string += str(item) + ","

    print("Dropping the following", len(unique_symbols), "stocks:")
    print("-" * 58)
    print(stock_drop_string, "\n")

    # drop symbols from df
    df.drop(nan_column_names, axis=1, inplace=True)

    # find remaining symbols after nan drop
    remaining_symbols = [df.columns.values[i].split(".")[1] for i in range(len(df.columns))]
    unique_remaining_symbols = np.unique(remaining_symbols)

    # create string of symbols remaining for user output
    stock_remaining_string = ""
    for item in unique_remaining_symbols:
        stock_remaining_string += str(item) + ","

    print("Keeping the following", len(unique_remaining_symbols), "stocks:")
    print("-" * 58)
    print(stock_remaining_string, "\n")

    return df


def preprocess(df, nan_tolerance_threshold=0.1):

    # update column names and index
    datetime = "Datetime"
    df.rename(columns={df.columns[0]: datetime}, inplace=True)
    df.set_index(datetime, inplace=True)

    # run preprocessing functions
    df = prepare_dat(df)
    df = remove_nan_columns(nan_tolerance_threshold, df)

    return df
