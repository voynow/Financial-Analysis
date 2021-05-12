import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_pipeline import run_pipeline

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, SimpleRNN, GRU, LSTM

import time


def get_data_by_feature(data, feature):
    col_names = data.columns

    return df[col_names[[col.split(".")[0] == feature for col in col_names]]]


def normalize(x):
    return (2 * (x - np.min(x)) / (np.max(x) - np.min(x))) - 1


def feature_target_split(price_data, lag):
    features = np.zeros((price_data.shape[0] - lag, lag + 1))
    target = np.zeros(price_data.shape[0] - lag)

    price_data = normalize(price_data)

    for i in range(lag, price_data.shape[0]):
        price_features = price_data[i - lag:i]
        features[i - lag] = np.hstack((price_features, np.mean(price_features)))
        target[i - lag] = (price_data[i] > 0) * 1

    return features, target

def prep_data(df):
    df_open = get_data_by_feature(df, "Open")
    df_close = get_data_by_feature(df, "Close")

    open_stocks = np.array([df_open.columns[i].split(".")[1] for i in range(len(df_open.columns))])
    close_stocks = np.array([df_close.columns[i].split(".")[1] for i in range(len(df_close.columns))])

    if np.array_equal(open_stocks, close_stocks):
        print("Test Passed: attributes contain matching stocks")
    else:
        print("ERROR: mismatched stocks exist between open_df and close_df")

    column_space = len(df_open.columns)
    columns = [df_open.columns[i].split('.')[1] for i in range(column_space)]

    price_change_data = np.array([df_close[df_close.columns[i]].values - df_open[df_open.columns[i]].values for i in range(column_space)])
    price_change_df = pd.DataFrame(price_change_data.T, columns=columns)
    
    x = []
    y = []
    for i in range(column_space):
        data_split = feature_target_split(price_change_data[i], 50)
        x.append(data_split[0])
        y.append(data_split[1])

    x = np.array(x)
    y = np.array(y)

    x = np.concatenate([x[i] for i in range(x.shape[0])])
    y = np.concatenate([y[i] for i in range(y.shape[0])]).reshape(x.shape[0], 1)
    return x, y

def build_model():

    model = Sequential()

    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['acc'])

    return model

def build_cnn():
    model = Sequential()
    model.add(Conv1D(32, 3, activation='relu'))
    model.add(Conv1D(32, 6, activation='relu'))
    model.add(Conv1D(32, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['acc'])
    return model

def build_rnn():
    model = Sequential()
    model.add(LSTM(128, return_sequences=True))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['acc'])
    return model


dir_path = r"../Data/1wk1m_0.csv"
dir_list = ["../Data/1wk1m_0.csv", "../Data/1wk1m_1.csv", "../Data/1wk1m_2.csv"]

# timeseries data for subset of Russ3000 stocks
df = run_pipeline(dir_list)

x_train, y_train = prep_data(df)

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

df = run_pipeline("../Data/1wk1m_5.csv")
x_test, y_test = prep_data(df)

y_train = y_train.flatten()
y_test = y_test.flatten()

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# start = time.time()
# build_model().fit(x_train, y_train, batch_size=4096, epochs=10, validation_data=(x_test, y_test))
# end = time.time()
# print(end-start)

# start = time.time()
# random_forest = RandomForestClassifier(n_jobs=-1)
# random_forest.fit(x_train, y_train)
# pred = random_forest.predict(x_test)
# end = time.time()
# print(accuracy_score(y_test, pred))
# print(end-start)

x_train = x_train[:, :, np.newaxis]
x_test = x_test[:, :, np.newaxis]
start = time.time()
build_cnn().fit(x_train, y_train, batch_size=4096, epochs=10, validation_data=(x_test, y_test))
end = time.time()
print(end-start)

# x_train = x_train[:, :, np.newaxis]
# x_test = x_test[:, :, np.newaxis]
# start = time.time()
# model = build_rnn()
# model.fit(x_train, y_train, batch_size=4096, epochs=10, validation_data=(x_test, y_test)) 
# model.predict(x_test)
# end = time.time()
# print(end-start)