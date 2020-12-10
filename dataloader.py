# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 17:12:40 2020

@author: voyno
"""

import os
import pandas as pd
import yfinance as yf


def main():

    # get stock symbols
    filename_symbols = 'Russell3000_symbols.csv'
    snp = pd.read_csv(filename_symbols, header=None)

    # create string of symbols
    symbol_string = ""
    snp_symbols = snp.values
    for i in range(len(snp_symbols)):
        symbol_string += snp_symbols[i] + " "

    symbol_string += "VOO "

    # download data from symbol string
    data = yf.download(tickers=symbol_string[0],
                       period="1wk",
                       interval="1m")

    # create filename
    week_count = 0
    folder = "Data/"
    filename_base = "1wk1m_"
    listdir = os.listdir(folder)

    for file in listdir:
        if filename_base + str(week_count) in file:
            week_count += 1

    # export data
    data.to_csv(folder + filename_base + str(week_count) + ".csv")


main()
