import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np

def training_data(window, lookback, train):
    """
    Prepares training data for the LSTM model by creating windows of consecutive samples from the data.
    
    params:
        window (int): Number of steps to predict
        lookback (int): Number of steps used for prediction
        train (string): Directory of training dataset
        
    returns:
        X_train, Y_train (array): Windowed training data
    """
    
    X_train = []
    Y_train = []
    train_df = pd.read_csv(train, index_col=[0])
    
    for i in range(len(train_df)-window-lookback):
        tt = train_df.values[i:i+lookback]
        X_train.append([[t] for t in tt])
        Y_train.append(train_df.values[lookback+i:lookback+i+window])
        
    X_train = np.array(X_train)
    Y_train = np.array(Y_train, dtype=object).astype(np.float32)

    return X_train, Y_train
    
class TS_predictor():
    
    def __init__(self, lstm_cell, window, lookback, optimizer, loss):
        
        self.window = window
        self.lookback = lookback
        self.lstm_model = Sequential()
        self.lstm_model.add(LSTM(lstm_cell, return_sequences=False))
        self.lstm_model.add(Dense(window))
        self.lstm_model.compile(optimizer=optimizer, loss=loss)
        
        
    def train(self, epochs, batch_size, dir_train, dir_model):
        """
        Trains the LSTM model and saves it in dir_model directory.
        
        params:
            epoch (int)
            batch_size (int)
            dir_train (string): Directory of training dataset
        """
        
        X_train, Y_train = training_data(self.window, self.lookback, dir_train)    
        self.lstm_model.fit(np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)), Y_train, batch_size=batch_size, epochs=epochs)
        self.lstm_model.save(dir_model)
        
    def predict(self, X_test_dir, model):
        """
        Loads LSTM model of your choice and predicts next n-steps in X_test time series.
        
        params:
            X_test_dir (string): Directory of test data
            model (string): Directory of model
            
        returns:
            predictions (array): Array of the next n-steps in the time series
        """
        
        TS_model = tf.keras.models.load_model(model)
        X_test = pd.read_csv(X_test_dir, index_col=[0])
        X_test = np.reshape(X_test.values, (1, self.lookback))
        predictions = TS_model.predict(X_test)
        return predictions
