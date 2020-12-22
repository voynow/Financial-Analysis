# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 20:55:53 2020

@author: voyno
"""

import pandas as pd
import numpy as np


def prepare_dat(df):

    for col in df.columns:
        if "." not in col:
            df.rename(columns={col: col + ".0"}, inplace=True)

    key_value_data = np.vstack([df.loc[df.index[0]].values, df.columns]).T
    lookup_dict = pd.DataFrame(key_value_data, columns=['symbols', 'columns'])

    print(df.head())

    df.drop([df.index[i] for i in [0, 1]], axis=0, inplace=True)
    df = df.astype('float')

    return df, lookup_dict


def symbol_to_columns(lookup_dict, symbols):
    """
    PARAMS
    lookup_dict : dataframe columns=[symbol, column] for lookup
    symbols     : list of stock symbols (all must be of type 'str')
    """
    columns_each_symbol = [np.where(lookup_dict['symbols'] == symbol)[0] for symbol in symbols]
    columns_indexes = np.array(columns_each_symbol).flatten()

    """
    RETURNS
    columns_names : names of columns associate to given symbol(s)
    """
    return lookup_dict['columns'].values[columns_indexes]


def columns_to_symbols(lookup_dict, columns):

    indexes = [np.where(lookup_dict['columns'] == column)[0][0] for column in columns]
    associated_symbols = lookup_dict['symbols'].iloc[indexes]

    return np.unique(associated_symbols)


def remove_nan_columns(threshold, df, lookup_dict):

    threshold_percentage = threshold * df.shape[0]

    print()
    print("*" * 58 + "\n\tProcessing series consisting of NAN values\n" + "*" * 58, "\n")
    nan_matrix = np.array([np.sum(np.isnan(df[column].values)) for column in df.columns])
    nan_columns = np.where(nan_matrix > threshold_percentage)[0]

    symbols_with_nan_cols = lookup_dict['symbols'].iloc[nan_columns]
    unique_symbols = np.unique(symbols_with_nan_cols)

    nan_symbol_positions = [np.where(symbols_with_nan_cols == symbol)[0] for symbol in unique_symbols]
    nan_symbol_column_count = [np.array(item).shape[0] for item in nan_symbol_positions]

    if nan_symbol_column_count != [6] * len(nan_symbol_column_count):
        print("Warning: NAN columns found, but there exists inconsistencies in NAN columns.")

    stock_drop_string = ""
    for item in unique_symbols:
        stock_drop_string += str(item) + ","

    print("Dropping the following", len(unique_symbols), "stocks:")
    print("-" * 58)
    print(stock_drop_string, "\n")
    df.drop(symbol_to_columns(lookup_dict, unique_symbols), axis=1, inplace=True)

    remaining_symbols = columns_to_symbols(lookup_dict, df.columns.values)
    stock_remaining_string = ""
    for item in remaining_symbols:
        stock_remaining_string += str(item) + ","

    print("Keeping the following", len(remaining_symbols), "stocks:")
    print("-" * 58)
    print(stock_remaining_string, "\n")

    return df


def preprocess(df, nan_tolerance_threshold=0.1):

    datetime = "Datetime"
    df.rename(columns={df.columns[0]: datetime}, inplace=True)
    df.set_index(datetime, inplace=True)

    df, lookup_dict = prepare_dat(df)
    df = remove_nan_columns(nan_tolerance_threshold, df, lookup_dict)

    return df, lookup_dict
