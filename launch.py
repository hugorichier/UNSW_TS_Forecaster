import argparse
import forecasting
import pandas as pd

def main(lstm_cell, window, lookback, optimizer, loss, epochs, batch_size, dir_train, dir_model, X_test, train):

    
    forecaster = forecasting.TS_predictor(lstm_cell = lstm_cell,
                                  window = window,
                                  lookback = lookback,
                                  optimizer = optimizer,
                                  loss = loss
                                 )
    
    if train == 1:
        forecaster.train(epochs = epochs,
                         batch_size = batch_size,
                         dir_train = dir_train,
                         dir_model = dir_model
                        )
    
    forecaster.predict(X_test_dir = X_test,
                       model = dir_model
                      )



parser = argparse.ArgumentParser()
parser.add_argument('--lstm_cell', action="store", help='Provides nb of lstm cells', type=int)
parser.add_argument('--window', action="store", help='Provides nb of steps the model predicts', type=int)
parser.add_argument('--lookback', action="store", help='Provides nb of steps for lookback period', type=int)
parser.add_argument('--optimizer', action="store", help='Provides model optimizer')
parser.add_argument('--loss', action="store", help='Provides loss function for model')
parser.add_argument('--epochs', action="store", help='Provides nb of epochs for training', type=int)
parser.add_argument('--batch_size', action="store", help='Provides batch size for training', type=int)
parser.add_argument('--dir_train', action="store", help='Provides directory for training data')
parser.add_argument('--dir_model', action="store", help='Provides directory for model')
parser.add_argument('--X_test', action="store", help='Data used for a prediction')
parser.add_argument('--train', action="store", help='1 if you want to train a model, else 0', type=int)

args = parser.parse_args()
main(args.lstm_cell, args.window, args.lookback, args.optimizer, args.loss, args.epochs, args.batch_size, args.dir_train, args.dir_model, args.X_test, args.train)
