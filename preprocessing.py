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

    columns_symbols = df.loc[df.index[0]].values
    df.columns = [df.columns[i].split(".")[0] + "." + columns_symbols[i] for i in range(len(df.columns))]

    df.drop([df.index[i] for i in [0, 1]], axis=0, inplace=True)
    df = df.astype('float')

    return df


def columns_to_symbols(lookup_dict, columns):

    indexes = [np.where(lookup_dict['columns'] == column)[0][0] for column in columns]
    associated_symbols = lookup_dict['symbols'].iloc[indexes]

    return np.unique(associated_symbols)


def remove_nan_columns(threshold, df):

    threshold_percentage = threshold * df.shape[0]

    print()
    print("*" * 58 + "\n\tProcessing series consisting of NAN values\n" + "*" * 58, "\n")
    nan_matrix = np.array([np.sum(np.isnan(df[column].values)) for column in df.columns])
    nan_columns = np.where(nan_matrix > threshold_percentage)[0]
    nan_column_names = df.columns[nan_columns]

    symbols_with_nan_cols = [nan_column_names[i].split(".")[1] for i in range(len(nan_column_names))]
    unique_symbols = np.unique(symbols_with_nan_cols)

    stock_drop_string = ""
    for item in unique_symbols:
        stock_drop_string += str(item) + ","

    print("Dropping the following", len(unique_symbols), "stocks:")
    print("-" * 58)
    print(stock_drop_string, "\n")
    df.drop(nan_column_names, axis=1, inplace=True)

    remaining_symbols = np.unique([df.columns.values[i].split(".")[1] for i in range(len(df.columns))])
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

    df = prepare_dat(df)
    df = remove_nan_columns(nan_tolerance_threshold, df)

    return df
