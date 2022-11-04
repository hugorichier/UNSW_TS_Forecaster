import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
import datetime
class TS_predictor():
    
    def __init__(self, lstm_cell, window, lookback, optimizer, loss):
        
        self.window = window
        self.lookback = lookback
        self.lstm_model = Sequential()
        self.lstm_model.add(LSTM(lstm_cell, return_sequences=False))
        self.lstm_model.add(Dense(window))
        self.lstm_model.compile(optimizer=optimizer, loss=loss)
        
        
    def train(self, epochs, batch_size, training_data):
        """
        Trains the LSTM model and saves it in dir_model directory.
        
        params:
            epoch (int)
            batch_size (int)
            dir_train (string): Directory of training dataset
        """
        X_train, Y_train = training_data
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        return self.lstm_model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_split=0.3, callbacks=[tensorboard_callback])
        
    def predict(self, testing_data):
        """
        Loads LSTM model of your choice and predicts next n-steps in X_test time series.
        
        params:
            X_test_dir (string): Directory of test data
            
        returns:
            predictions (array): Array of the next n-steps in the time series
        """
        
        X_test, Y_test = testing_data 
        predictions = self.lstm_model.predict(X_test)
        return predictions