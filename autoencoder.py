import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

np.random.seed(1)
tf.random.set_seed(1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed, Input

import yfinance as yf

from data_pipeline import run_pipeline

HIDDEN_UNITS = 50
TRAIN_SIZE = 1750
TEST_SIZE = 200
BATCH_SIZE = 64
TRAIN_SAMPLES = 250000
EPOCHS = 10
LAG = 50

def get_data_by_feature(data, feature):
    col_names = data.columns

    return df[col_names[[col.split(".")[0] == feature for col in col_names]]]

def normalize(x):
    return (2 * (x - np.min(x)) / (np.max(x) - np.min(x))) - 1

def feature_target_split(price_data, lag):
    features = np.zeros((price_data.shape[0] - lag, lag))
    target = np.zeros(price_data.shape[0] - lag)

    price_data = normalize(price_data)

    for i in range(lag, price_data.shape[0]):
        features[i - lag] = price_data[i - lag:i]
        #features[i - lag] = np.hstack((price_features, np.mean(price_features)))
        target[i - lag] = (price_data[i] > 0) * 1
    
    return features, target

def build_model_lstm(lag):

    model = Sequential()

    model.add(LSTM(HIDDEN_UNITS, input_shape=(lag,1), activation='relu'))
    model.add(RepeatVector(lag))
    model.add(LSTM(HIDDEN_UNITS, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(1)))

    model.compile(optimizer="adam", loss="mae", metrics=[tf.keras.metrics.MeanAbsoluteError()])
    
    model.summary()

    return model

def build_model_mlp(lag):

    model = Sequential()

    model.add(Input(shape=(lag)))
    model.add(Dense(30))
    model.add(Dense(lag))

    model.compile(optimizer="adam", loss="mae", metrics=[tf.keras.metrics.MeanAbsoluteError()])
    
    model.summary()

    return model

def sample_test_sequences(data, num_samples, lag):
    
    sample = np.zeros((num_samples, LAG))
    
    # Data should be a numpy array
    for i in range(num_samples):
        rand_num = int(np.random.uniform(0, data.shape[0]-lag))
        sample[i] = data[rand_num:rand_num + lag][0]
        
    return sample

def plot_comparisons(model, test_samples): 
    fig, axs = plt.subplots(6,2, figsize=(10,8), sharex=True, sharey=True)
    axs[0, 0].set_title("Model reconstruction")
    axs[0, 1].set_title("True time series")
    axs[5, 0].set_xlabel("Timesteps")
    axs[5, 1].set_xlabel("Timesteps")


    for j in range(6):
        # Plot model prediction on never seen before test data
        pred = model.predict(test_samples[j].reshape(1,test_samples.shape[1]))
        axs[j, 0].plot(pred[0])

        # Plot the true test data
        axs[j, 1].plot(test_samples[j])
        
# using first week csv for now
dir_path = r"Data/1wk1m_0.csv"

# timeseries data for subset of Russ3000 stocks
df = run_pipeline(dir_path)
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

X_train = np.zeros((column_space, TRAIN_SIZE - LAG, LAG))
X_test = np.zeros((column_space, TEST_SIZE - LAG, LAG))
y_train = np.zeros((column_space, TRAIN_SIZE - LAG))
y_test = np.zeros((column_space, TEST_SIZE - LAG))

train = price_change_df[:TRAIN_SIZE].T.values
test = price_change_df[-TEST_SIZE:].T.values

for i in range(column_space):
    data_split = feature_target_split(train[i], LAG)
    X_train[i] = data_split[0]
    y_train[i] = data_split[1]
    
    data_split = feature_target_split(test[i], LAG)
    X_test[i] = data_split[0]
    y_test[i] = data_split[1]

X_train = np.concatenate([X_train[i] for i in range(X_train.shape[0])])
X_test = np.concatenate([X_test[i] for i in range(X_test.shape[0])])

y_train = np.concatenate([y_train[i] for i in range(y_train.shape[0])]).reshape(X_train.shape[0], 1)
y_test = np.concatenate([y_test[i] for i in range(y_test.shape[0])]).reshape(X_test.shape[0], 1)

print(f"Train data shape: {X_train.shape}")
print(f"Train labels shape: {y_train.shape}")
print(f"Test data shape:  {X_test.shape}")  
print(f"Test labels shape:  {y_test.shape}")

train_samples = X_train[:TRAIN_SAMPLES]
test_samples = sample_test_sequences(X_test, 10, LAG)
train_samples.shape, test_samples.shape

model = build_model_mlp(LAG)

history = model.fit(train_samples, 
                    train_samples, 
                    epochs=10, 
                    batch_size=BATCH_SIZE)

# Plot the model reconstructing never seen before test series
plt.plot(model.predict(X_test[0].reshape(1, LAG))[0], label="True series")
plt.plot(X_test[0], label="Reconstruction");
plt.legend();
plt.savefig("test_reconstruction.png")
        
plot_comparisons(model, test_samples)

# Build predictive model

# Freeze the hidden dense layer
model.trainable = False

# connect the encoder to the output layer
inputs = model.layers[0].output
x = Dense(1, activation='sigmoid')(inputs)
predictive_model = keras.models.Model(inputs=model.inputs, outputs=x)

predictive_model.summary()

# Compile and train predictive model
predictive_model.compile(optimizer="adam", loss="BinaryCrossentropy", metrics=['accuracy'])

history_pred = predictive_model.fit(X_train, 
                                    y_train,
                                   epochs=EPOCHS,
                                   batch_size=BATCH_SIZE,
                                    validation_split=0.2,                            
                                   )

# 5000 is used for time considerations. The .predict call takes awhile
preds = [predictive_model.predict(X_test[i].reshape(1,50)) for i in range(5000)]

# Looks at each prediction: predicts 0 if value is closer to 1 (which means greater prob that class 0 is correct) 
# and predicts 1 if value is closer to 0, as class 0 is less likely to be correct
adj_preds = []
for pred in preds:
    if pred > 0.5:
        adj_preds.append(0)
    else:
        adj_preds.append(1)
        
print(f"Accuracy on test data: {accuracy_score(adj_preds, y_test[:5000])}")
