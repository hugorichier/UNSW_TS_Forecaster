import pandas as pd
import numpy as np
import optuna

def train_test_split(data):
    """
    Split data into training and testing sets.
    
    params:
        data (array)
    returns:
        train_df, test_df (array): Splits of dataset
    """
    
    n = len(data)
    train_df = data[0:int(n*0.7)]
    test_df = data[int(n*0.7):]
    return train_df, test_df

def preprocessing(df, feature, resample, norm, weekend):
    """
    Preprocess time series data for better predicitons.
    
    params:
        df (pd.DatFrame): Raw data
        feature (String): Feature on whcich preprocessing is done
        resample (string): Timeframe on which to resample data
        
    returns:
        df_t (array): Transformed data
    """
    
    df['ts'] = pd.to_datetime(df['ts'])
    df = df.resample(resample, on='ts').mean()
    df = df.dropna()
    df_t = df[feature]
    if weekend == 0:
        df_t = df_t[df_t.index.weekday<5] # remove weekends
    
    if norm == 1:
        m = df_t.mean()
        s = df_t.std()
        df_t = (df_t-m)/s # Z-score normalization
        
    return df_t


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
    train_df = train
    
    for i in range(len(train_df)-window-lookback):
        tt = train_df.values[i:i+lookback]
        X_train.append([[t] for t in tt])
        Y_train.append(train_df.values[lookback+i:lookback+i+window])
        
    X_train = np.array(X_train)
    Y_train = np.array(Y_train, dtype=object).astype(np.float32)

    return X_train, Y_train


def testing_data(window, lookback, test):
    """
    Prepares training data for the LSTM model by creating windows of consecutive samples from the data.
    
    params:
        window (int): Number of steps to predict
        lookback (int): Number of steps used for prediction
        train (string): Directory of training dataset
        
    returns:
        X_train, Y_train (array): Windowed training data
    """
    
    X_test = []
    Y_test = []
    test_df = test
    
    for i in range(len(test_df)-window-lookback):
        tt = test_df.values[i:i+lookback]
        X_test.append([[t] for t in tt])
        Y_test.append(test_df.values[lookback+i:lookback+i+window])
        
    X_test = np.array(X_test)
    Y_test = np.array(Y_test, dtype=object).astype(np.float32)

    return X_test, Y_test