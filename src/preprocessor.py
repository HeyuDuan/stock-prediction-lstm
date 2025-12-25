import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
import os

class StockPreprocessor:
    """Preprocessor for Stock Price Time Series Data (LSTM Input Preparation)"""
    
    def __init__(self, lookback_days=60, feature_columns=['Close']):
        self.lookback_days = lookback_days
        self.feature_columns = feature_columns
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def load_and_prepare_data(self, data_path, train_split=0.8):
        """
        Loads, preprocesses and splits stock data for LSTM training
        
        Parameters:
            data_path (str): Path to CSV file containing stock data
            train_split (float): Ratio of training data (default: 0.8)
            
        Returns:
            tuple: (X_train, y_train, X_test, y_test, original_df)
            - X_train: Training input sequences (shape: [samples, lookback_days, features])
            - y_train: Training target values (closing prices)
            - X_test: Test input sequences
            - y_test: Test target values
            - original_df: Original unprocessed DataFrame (for visualization/analysis)
        """
        print("Loading data...")
        df = pd.read_csv(data_path, parse_dates=['Date'])
        
        # Ensure data is sorted by date
        df = df.sort_values('Date')
        
        # Extract feature columns
        feature_data = df[self.feature_columns].values
        
        # Normalize data to [0,1] range
        print("Normalizing data...")
        scaled_data = self.scaler.fit_transform(feature_data)
        
        # Create time series sequences for LSTM input
        print("Creating time series sequences...")
        X, y = self._create_sequences(scaled_data)
        
        # Split dataset into train/test sets
        split_idx = int(len(X) * train_split)
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_test, y_test = X[split_idx:], y[split_idx:]
        
        print(f"Data preparation completed!")
        print(f"  Training set: X={X_train.shape}, y={y_train.shape}")
        print(f"  Test set: X={X_test.shape}, y={y_test.shape}")
        
        return X_train, y_train, X_test, y_test, df
    
    def _create_sequences(self, data):
        """Creates input/output sequences for LSTM time series prediction
        
        Parameters:
            data (np.array): Normalized feature data (shape: [timesteps, features])
            
        Returns:
            tuple: (X, y) where:
            - X: Input sequences (shape: [samples, lookback_days, features])
            - y: Target values (next time step closing price)
        """
        X, y = [], []
        
        for i in range(self.lookback_days, len(data)):
            X.append(data[i-self.lookback_days:i])
            y.append(data[i, 0])  # Predict closing price for next time step
        
        return np.array(X), np.array(y)
    
    def save_scaler(self, filepath):
        """Saves MinMaxScaler to pickle file for future inverse transformation
        
        Parameters:
            filepath (str): Path to save the scaler object
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Scaler saved to: {filepath}")
    
    def load_scaler(self, filepath):
        """Loads pre-saved MinMaxScaler from pickle file
        
        Parameters:
            filepath (str): Path to the scaler pickle file
            
        Returns:
            MinMaxScaler: Loaded scaler object
        """
        with open(filepath, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"Scaler loaded from {filepath}")
        return self.scaler
    
    def inverse_transform(self, scaled_data):
        """Inverse transforms normalized data back to original price range
        
        Parameters:
            scaled_data (np.array): Normalized data (range [0,1])
            
        Returns:
            np.array: Data in original price range
        """
        return self.scaler.inverse_transform(scaled_data)