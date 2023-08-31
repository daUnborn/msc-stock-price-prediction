#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 18:01:08 2023

@author: camaike
"""

'''
Install keras_tuner to ensure hyperparameter optimization works as expected.

For pyquant, check installation documentation here - https://pyquantnews.com/wp-content/uploads/2023/03/How-To-Set-Up-the-Python-Quant-Stack-in-a-Custom-Quant-Lab.pdf


'''

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, GRU, Bidirectional
from google.colab import drive
#39drive.mount('/content/drive')
import keras_tuner
from tensorflow import keras
from tensorflow.keras import models, layers, optimizers, losses, callbacks
from tensorflow.keras.layers import Dropout

def prepare_data(stock, start, end, lookback):

    df = yf.download(stock, start, end, progress=False, auto_adjust=False)

    df['Stock'] = stock
    df['returns'] = df['Adj Close'].pct_change()
    df['momentum'] = df['returns'].rolling(lookback).mean().shift(1)
    df['volatility'] = df['returns'].rolling(lookback).std().shift(1)
    df['distance'] = (df['Adj Close'] - df['Adj Close'].shift(1))
    df['lag1'] = df['returns'].shift(1) #the parameter in ranges from 1 - 5
    df['lag2'] = df['returns'].shift(2)
    df['lag3'] = df['returns'].shift(3)
    df['lag4'] = df['returns'].shift(4)
    df['lag5'] = df['returns'].shift(5)
    df['expected_return'] = df['returns'].rolling(window=lookback).mean()
    df['actual_return'] = df['returns'].shift(1)
    df['momentum_anomaly'] = ((df['actual_return'] - df['expected_return']) / df['expected_return']
                              .rolling(window=lookback).mean())

    df.dropna(inplace=True)

    return df

def prepare_multiple_stock_data(stock_list, start, end):

    combined_df = pd.DataFrame()

    for s in stock_list:
      df = prepare_data(s, start, end, lookback)
      combined_df = pd.concat([combined_df, df], axis=0)

    return combined_df


def normalize_split_data(df, lookback, train=True):
    features = ['momentum', 'volatility', 'distance', 'lag1', 'lag2', 'lag3', 'lag4', 'lag5', 'actual_return', 'expected_return', 'momentum_anomaly']
    label = 'returns'

    X = df[features].values
    y = df[label].values

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    X_sequences = []
    y_sequences = []

    for i in range(len(X_scaled) - lookback + 1):
        X_sequences.append(X_scaled[i:i+lookback])
        y_sequences.append(y[i+lookback-1])

    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)

    if train:
        X_train, X_val, y_train, y_val = train_test_split(X_sequences, y_sequences, test_size=0.20, shuffle=True)
        return X_train, X_val, y_train, y_val
    else:
        X_test = X_sequences
        y_test = y_sequences
        return X_test, y_test

def create_lstm_model(X_train, X_val, y_train, y_val):
    model = Sequential(name='LSTM')
    model.add(LSTM(32, input_shape=(lookback, len(features))))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    early_Stop = callbacks.EarlyStopping(monitor='val_mse', patience=3)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=64, verbose=0, callbacks=[early_Stop])
    return model, history

def create_gru_model(X_train, X_val, y_train, y_val):

    model = Sequential(name='GRU')
    model.add(GRU(32, input_shape=(lookback, len(features))))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    early_Stop = callbacks.EarlyStopping(monitor='val_mse', patience=3)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=64, verbose=0, callbacks=[early_Stop])
    return model, history

def create_bi_lstm(X_train, X_val, y_train, y_val):
    model = Sequential(name='Bidirectional')
    # Add a bidirectional LSTM layer
    model.add(Bidirectional(LSTM(units=64, return_sequences=True, activation='tanh',
                                 dropout=0.3), input_shape=(lookback, len(features))))
    model.add(LSTM(units=32, return_sequences=True))
    model.add(LSTM(units=32))
    # Add a dense layer with tunable hyperparameters
    model.add(Dense(units=32, activation='tanh'))
    model.add(Dropout(0.1))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    early_Stop = callbacks.EarlyStopping(monitor='val_mse', patience=3)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=50, batch_size=64, verbose=0, callbacks=[early_Stop])
    return model, history

def check_outliers(df, stock):
    df_1 = df.copy()

    df_1 = df_1.loc[df['Stock'] == stock]
    df_rolling = df_1[["returns"]].rolling(window=30).agg(["mean", "std"])
    df_rolling.columns = df_rolling.columns.droplevel()

    df_1 = df_1.join(df_rolling)

    N_SIGMAS = 3
    df_1["upper"] = df_1["mean"] + N_SIGMAS * df_1["std"]
    df_1["lower"] = df_1["mean"] - N_SIGMAS * df_1["std"]

    df_1["outlier"] = ((df_1["returns"] > df_1["upper"]) | (df_1["returns"] < df_1["lower"]))

    fig, ax = plt.subplots()
    df_1[["returns", "upper", "lower"]].plot(ax=ax)
    ax.scatter(df_1.loc[df_1["outlier"]].index, df_1.loc[df_1["outlier"], "returns"], color="black", label="outlier")
    ax.set_title(f"{stock}'s Adj Close Outliers")
    ax.legend()
    #plt.savefig('/content/drive/MyDrive/ml-group-3/images/outliers.png')
    plt.show()

def plot_close_return(df, stock, columns, title, ylabel):
    df_1 = df.copy()

    df_1 = df_1.loc[df['Stock'] == stock]
    df_1[columns].plot(subplots=True, sharex=True, title=title, ylabel=ylabel)
    #plt.savefig('/content/drive/MyDrive/ml-group-3/images/returns.png')

def plot_performance(data, key, label, xlabel, ylabel, title):
  plt.plot(data[key], label=label)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.title(title)
  plt.legend()

def cal_kur_skew(df, column, stock_list):
    stat_df = pd.DataFrame(index=stock_list, columns=['skewness', 'kurtosis'])

    for s in stock_list:

      skewness = skew(df.loc[df.Stock == s][column])
      kurt = kurtosis(df.loc[df.Stock == s][column])

      stat_df.loc[s, 'skewness'] = skewness
      stat_df.loc[s, 'kurtosis'] = kurt

    return stat_df

def evaluate_model(model, result, X_val, y_val):

    y_pred = model.predict(X_val, verbose=0)

    if result == 'metrics':
        evaluation_outputs = {}

        # Calculate MAE
        mae = mean_absolute_error(y_val, y_pred)

        # Calculate RMSE
        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, y_pred)
        f1 = f1_score(y_val > 0, y_pred > 0)
        cm = confusion_matrix(y_val > 0, y_pred > 0)
        loss = model.evaluate(X_val, y_val, verbose=0)

        evaluation_outputs[model.name] = {
            'mse':mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'f1': f1
        }

        evaluation_df = pd.DataFrame.from_dict(evaluation_outputs, orient='index')
        return evaluation_df
    else:
        actual_cumulative_return = ((1 + y_val).cumprod() - 1) * 100
        predicted_cumulative_return = ((1 + y_pred).cumprod() - 1) * 100

        return actual_cumulative_return, predicted_cumulative_return

stock_list = ['BP.L','HSBA.L','SHEL.L','ULVR.L','VOD.L','BHP.L','AZN.L','GSK.L','LLOY.L','TUI.L', 'RR.L', 'IHG.L']
train_stock = 'BP.L'
start = '2012-01-01'
end = '2022-12-31'
lookback = 2
features = ['momentum', 'volatility', 'distance', 'lag1', 'lag2', 'lag3', 'lag4', 'lag5', 'actual_return', 'expected_return', 'momentum_anomaly']


df = prepare_data('BP.L', start, end, lookback)
print(df.shape)
X_train, X_val, y_train, y_val = normalize_split_data(df, lookback, train=True)
m_df = prepare_multiple_stock_data(stock_list, start, end)
print(m_df.shape)

#plot the returns of BP.L
plot_close_return(df, 'BP.L', 'returns', f'Returns of BP.L from {start} to {end}', 'returns')
plt.show()

#plot the cumulative returns of BP.L
df['cumm'] = (1 + df['returns']).cumprod()
plot_close_return(df, 'BP.L', 'cumm', f'Cummulative Returns of BP.L from {start} to {end}', 'cumulative returns')
plt.show()

plot_close_return(df, 'BP.L', 'momentum', f'Momentum of BP.L from {start} to {end}', 'Momentum')
plt.show()

plot_close_return(df, 'BP.L', 'volatility', f'Volatility of BP.L from {start} to {end}', 'Volatility')
plt.show()
plot_close_return(df, 'BP.L', 'momentum_anomaly', f'Momentum Anomaly of BP.L from {start} to {end}', 'Momentum Anomaly')
plt.show()

plot_close_return(df, 'BP.L', 'momentum_anomaly', f'Momentum Anomaly of BP.L from {start} to {end}', 'Momentum Anomaly')
plt.show()

# @title Default title text
stat_df = cal_kur_skew(m_df, 'Adj Close', stock_list)
print(stat_df)

data = df['Adj Close']
data_skewness = skew(data)

plt.figure(figsize=(10, 6))

# Plot histogram of the data
plt.hist(data, bins=30, alpha=0.7, color='blue', edgecolor='black')

# Add a vertical line to indicate the mean
plt.axvline(data.mean(), color='red', linestyle='dashed', linewidth=1)

# Annotate the skewness value
plt.annotate(f"Skewness: {data_skewness:.2f}", xy=(0.05, 0.9), xycoords='axes fraction', fontsize=12)

plt.xlabel('Data')
plt.ylabel('Frequency')
plt.title('Histogram of Data and Skewness for BP.L')
plt.grid(True)
plt.show()

data_kurtosis = kurtosis(data)

plt.figure(figsize=(10, 6))

# Plot histogram of the data
plt.hist(data, bins=30, alpha=0.7, color='blue', edgecolor='black')

# Add a vertical line to indicate the mean
plt.axvline(data.mean(), color='red', linestyle='dashed', linewidth=1)

# Annotate the kurtosis value
plt.annotate(f"Kurtosis: {data_kurtosis:.2f}", xy=(0.05, 0.9), xycoords='axes fraction', fontsize=12)

plt.xlabel('Data')
plt.ylabel('Frequency')
plt.title('Histogram of Data and Kurtosis for BP.L')
plt.grid(True)
plt.show()

lstm_model, hist = create_lstm_model(X_train, X_val, y_train, y_val)

lstm_df = evaluate_model(lstm_model, 'metrics', X_val, y_val)
print(lstm_df)

plt.figure(figsize=(6, 5))
plot_performance(hist.history, 'loss', 'Train Loss', 'Epoch', 'Loss', 'LSTM Training and Validation Loss')
plot_performance(hist.history, 'val_loss', 'Validation Loss', 'Epoch', 'Loss', 'LSTM Training and Validation Loss')
plt.grid()
plt.show()

plt.figure(figsize=(6, 5))
plot_performance(hist.history, 'mse', 'Train mse', 'Epoch', 'mse', 'LSTM Training and Validation mse')
plot_performance(hist.history, 'val_mse', 'Validation mse', 'Epoch', 'mse', 'LSTM Training and Validation mse')
plt.grid()
plt.show()

gru_model, hist = create_gru_model(X_train, X_val, y_train, y_val)

gru_df = evaluate_model(gru_model, 'metrics', X_val, y_val)
print(gru_df)

plt.figure(figsize=(6, 5))
plot_performance(hist.history, 'loss', 'Train Loss', 'Epoch', 'Loss', 'GRU Training and Validation Loss')
plot_performance(hist.history, 'val_loss', 'Validation Loss', 'Epoch', 'Loss', 'GRU Training and Validation Loss')
plt.grid()
plt.show()

plt.figure(figsize=(6, 5))
plot_performance(hist.history, 'mse', 'Train mse', 'Epoch', 'mse', 'GRU Training and Validation mse')
plot_performance(hist.history, 'val_mse', 'Validation mse', 'Epoch', 'mse', 'GRU Training and Validation mse')
plt.grid()
plt.show()

bi_model, hist = create_bi_lstm(X_train, X_val, y_train, y_val)
bi_lstm_df = evaluate_model(bi_model, 'metrics', X_val, y_val)
print(bi_lstm_df)

plt.figure(figsize=(6, 5))
plot_performance(hist.history, 'loss', 'Train Loss', 'Epoch', 'Loss', 'Bi-LSTM Training and Validation Loss')
plot_performance(hist.history, 'val_loss', 'Validation Loss', 'Epoch', 'Loss', 'Bi-LSTM Training and Validation Loss')
plt.grid()
plt.show()

plt.figure(figsize=(6, 5))
plot_performance(hist.history, 'mse', 'Train mse', 'Epoch', 'mse', 'Bi-LSTM Training and Validation mse')
plot_performance(hist.history, 'val_mse', 'Validation mse', 'Epoch', 'mse', 'Bi-LSTM Training and Validation mse')
plt.grid()
plt.show()

# Define the advanced version of the function
def build_model(hp):
    model = Sequential(name='Bidirectional')

    # Define hyperparameters for Bidirectional LSTM layer
    units_lstm1 = hp.Int('units_lstm1', min_value=8, max_value=16, step=8)

    # Add a bidirectional LSTM layer with tunable hyperparameters
    model.add(Bidirectional(LSTM(units=units_lstm1, return_sequences=True),input_shape=(lookback, len(features))))

    # Define hyperparameters for additional LSTM layers
    units_lstm2 = hp.Int('units_lstm2', min_value=8, max_value=16, step=8)
    units_lstm3 = hp.Int('units_lstm3', min_value=8, max_value=16, step=8)

    # Add additional LSTM layers with tunable hyperparameters
    model.add(LSTM(units=units_lstm2, return_sequences=True))
    model.add(LSTM(units=units_lstm3))

    # Output layer
    model.add(Dense(1))

    # Compile the model with hyperparameter tuning
    optimizer = hp.Choice('optimizer', values=['adam', 'rmsprop'])
    loss = 'mse'
    metrics = ['mse']
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model

build_model(keras_tuner.HyperParameters())

#Initialize a random search on the model using the parameters identified

tuner = keras_tuner.RandomSearch(
    hypermodel=build_model,
    objective=keras_tuner.Objective("mse", direction="min"),
    max_trials=10,
    executions_per_trial=1
)

#Start the search
tuner.search(X_train, y_train, epochs=50, validation_data=(X_val, y_val))
best_model = tuner.get_best_models()[0]

tuner.results_summary()

def tunned_bi_lstm(X_train, y_train, X_val, y_val):
    model = Sequential(name='Tuned_Bi_LSTM')
    # Add a bidirectional LSTM layer
    model.add(Bidirectional(LSTM(units=16, return_sequences=True),
                            input_shape=(lookback, len(features))))

    model.add(LSTM(units=16, return_sequences=True))
    model.add(LSTM(units=16))

    # Output layer
    model.add(Dense(1))

    # Compile the model with hyperparameter tuning

    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    early_Stop = callbacks.EarlyStopping(monitor='val_mse', patience=3)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, verbose=0, callbacks=[early_Stop])

    return model, history

'''
Trial 10 Complete [00h 00m 49s]
mse: 1.4697322967549553e-06

Best mse So Far: 7.1809131441114e-07
Total elapsed time: 00h 09m 24s
Results summary
Results in ./untitled_project
Showing 10 best trials
Objective(name="mse", direction="min")

Trial 01 summary
Hyperparameters:
units_lstm1: 16
units_lstm2: 16
units_lstm3: 16
optimizer: adam
Score: 7.1809131441114e-07
'''

tuned_model, hist = tunned_bi_lstm(X_train, y_train, X_val, y_val)

bi_tuned_df = evaluate_model(tuned_model, 'metrics', X_val, y_val)
print(bi_tuned_df)

plt.figure(figsize=(6, 5))
plot_performance(hist.history, 'loss', 'Train Loss', 'Epoch', 'Loss', 'Tuned Bi-LSTM Training and Validation Loss')
plot_performance(hist.history, 'val_loss', 'Validation Loss', 'Epoch', 'Loss', 'Tuned Bi-LSTM Training and Validation Loss')
plt.grid()
plt.show()

plt.figure(figsize=(6, 5))
plot_performance(hist.history, 'mse', 'Train mse', 'Epoch', 'mse', 'Bi-Tuned Bi-LSTM Training and Validation mse')
plot_performance(hist.history, 'val_mse', 'Validation mse', 'Epoch', 'mse', 'Bi-Tuned Bi-LSTM Training and Validation mse')
plt.grid()
plt.show()

#1. Get one model trained by a single stock data. Test the model
# with stock data from the 10 different stocks

results_df = pd.DataFrame(columns=['Stock', 'Actual Return', 'Predicted Return'])
mse_s_df = pd.DataFrame(columns=['mse','mae','rmse','r2','f1'])

# Create a subplot for the charts
fig, axs = plt.subplots(4, 3, figsize=(10, 10))
axs = axs.ravel()

for idx, stock in enumerate(stock_list):
    stock_df = prepare_data(stock, '2023-01-01', '2023-03-31', lookback)
    X_test, y_test = normalize_split_data(stock_df, lookback, train=False)
    actual_return_s, predicted_return_s = evaluate_model(tuned_model, 'returns', X_test, y_test)

    tuned_model_s = evaluate_model(tuned_model, 'metrics', X_test, y_test)

    mse_s_df = pd.concat([mse_s_df, tuned_model_s], ignore_index=True)

    actual_return = actual_return_s[-1]
    predicted_return = predicted_return_s[-1]

    # Add the results to the DataFrame
    results_df.loc[idx] = [stock, actual_return, predicted_return]

    # Plot the predicted vs. actual returns in a subplot
    axs[idx].plot(predicted_return_s, label='Predicted')
    axs[idx].plot(actual_return_s, label='Actual')
    axs[idx].set_title(f'Predicted vs Actual for {stock}')
    axs[idx].grid()
    axs[idx].legend()

# Adjust layout spacing
plt.tight_layout()

# Show the DataFrame with results
print(results_df)
print('\n')
mse = mse_s_df.mse.mean()
print(f'{mse:.15f}')

# Show all the subplots
plt.show()

all_df = prepare_multiple_stock_data(stock_list, start, end)
X_train_all, X_val_all, y_train_all, y_val_all = normalize_split_data(all_df, lookback, train=True)

tuned_model_a, hist_a = tunned_bi_lstm(X_train_all, y_train_all, X_val_all, y_val_all)

#all against 1
# Create an empty DataFrame to store the results
results_df = pd.DataFrame(columns=['Stock', 'Actual Return', 'Predicted Return'])
mse_all_df = pd.DataFrame(columns=['mse','mae','rmse','r2','f1'])

# Create a subplot for the charts
fig, axs = plt.subplots(4, 3, figsize=(10, 10))
axs = axs.ravel()

for idx, stock in enumerate(stock_list):
    all_test_df = prepare_data(stock, '2023-01-01', '2023-03-31', lookback)
    X_test_all, y_test_all = normalize_split_data(all_test_df, lookback, train=False)

    actual_return_all, predicted_return_all = evaluate_model(tuned_model_a, 'returns', X_test_all, y_test_all)

    tuned_model_all = evaluate_model(tuned_model_a, 'metrics', X_test_all, y_test_all)

    mse_all_df = pd.concat([mse_all_df, tuned_model_all], ignore_index=True)

    actual_return = actual_return_all[-1]
    predicted_return = predicted_return_all[-1]

    # Add the results to the DataFrame
    results_df.loc[idx] = [stock, actual_return, predicted_return]

    # Plot the predicted vs. actual returns in a subplot
    axs[idx].plot(predicted_return_all, label='Predicted')
    axs[idx].plot(actual_return_all, label='Actual')
    axs[idx].set_title(f'Predicted vs Actual for {stock}')
    axs[idx].grid()
    axs[idx].legend()

# Adjust layout spacing
plt.tight_layout()

# Show the DataFrame with results
print(results_df)
print('\n')
mse_all = mse_all_df.mse.mean()
print(f'{mse_all:.15f}')

# Show all the subplots
plt.show()

#5 to 5

half_df = prepare_multiple_stock_data(stock_list[6:], start, end)
X_train_half, X_val_half, y_train_half, y_val_half = normalize_split_data(half_df, lookback, train=True)

tuned_model_half, hist_half = tunned_bi_lstm(X_train_half, y_train_half, X_val_half, y_val_half)

# Create an empty DataFrame to store the results
results_df = pd.DataFrame(columns=['Stock', 'Actual Return', 'Predicted Return'])
mse_df = pd.DataFrame(columns=['mse','mae','rmse','r2','f1'])

# Create a subplot for the charts
fig, axs = plt.subplots(3, 2, figsize=(10, 10))
axs = axs.ravel()

for idx, stock in enumerate(stock_list[:6]):
    half_test_df = prepare_data(stock, '2023-01-01', '2023-03-31', lookback)
    X_test_half, y_test_half = normalize_split_data(half_test_df, lookback, train=False)
    actual_return_, predicted_return_ = evaluate_model(tuned_model_half, 'returns', X_test_half, y_test_half)
    tuned_model_half_df = evaluate_model(tuned_model_half, 'metrics', X_test_half, y_test_half)

    mse_df = pd.concat([mse_df, tuned_model_half_df], ignore_index=True)

    actual_return = actual_return_[-1]
    predicted_return = predicted_return_[-1]

    # Add the results to the DataFrame
    results_df.loc[idx] = [stock, actual_return, predicted_return]

    # Plot the predicted vs. actual returns in a subplot
    axs[idx].plot(predicted_return_, label='Predicted')
    axs[idx].plot(actual_return_, label='Actual')
    axs[idx].set_title(f'Predicted vs Actual for {stock}')
    axs[idx].grid()
    axs[idx].legend()

# Adjust layout spacing
plt.tight_layout()

# Show the DataFrame with results
print(results_df)
print('\n')
mse_half = mse_df.mse.mean()
print(f'{mse_half:.15f}')

# Show all the subplots
plt.show()

momentum_stocks = ['TSCO.L',
'SKG.L',
'MRO.L',
'ITRK.L',
'CRH.L',
'CNA.L',
'SBRY.L',
'SMIN.L',
'SMDS.L',
'AAF.L',
'CPG.L',
'FLTR.L',
'MNDI.L',
'ADM.L',
'ABF.L',
'IMB.L',
'ABDN.L',
'BKG.L',
'WPP.L',
'RIO.L',
'RS1.L',
'CRDA.L',
'SN.L',
'FRAS.L',
'HSBA.L',
'FCIT.L',
'UTG.L',
'GLEN.L',
'WTB.L',
'BDEV.L'
]

# Create an empty DataFrame to store the results
momentum_ = pd.DataFrame(columns=['Stock', 'Actual Return', 'Predicted Return'])

for idx, stock in enumerate(momentum_stocks):
    momentum_df = prepare_data(stock, '2023-01-01', '2023-03-31', lookback)
    X_test_momentum, y_test_momentum = normalize_split_data(momentum_df, lookback, train=False)
    actual_return_momentum, predicted_return_momentum = evaluate_model(tuned_model_half, 'returns', X_test_momentum, y_test_momentum)

    actual_return = actual_return_momentum[-1]
    predicted_return = predicted_return_momentum[-1]

    # Add the results to the DataFrame
    momentum_.loc[idx] = [stock, actual_return, predicted_return]

momentum_ = momentum_.sort_values(by='Predicted Return', ascending=False)

total_length = momentum_.shape[0]
upper_split = total_length // 3
buttom_split = 2 * (total_length // 3)

overperforming_stocks = momentum_[['Stock','Predicted Return', 'Actual Return']][:upper_split]
underperforming_stocks = momentum_[['Stock','Predicted Return', 'Actual Return']][buttom_split:]


print("First 1/3:")
print(overperforming_stocks)

print("\nLast 1/3:")
print(underperforming_stocks)


