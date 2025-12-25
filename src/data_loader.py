import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class StockDataLoader:
    """Loads historical stock data from Yahoo Finance and calculates technical indicators"""
    
    def __init__(self, symbol, start_date, end_date):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
    
    def download_data(self, save_path=None):
        """Downloads historical stock data from Yahoo Finance
        
        Parameters:
            save_path (str, optional): Path to save CSV file (default: None)
            
        Returns:
            pd.DataFrame: Historical stock data with Date/Open/High/Low/Close/Volume columns
            
        Raises:
            ValueError: If no data is downloaded for the given symbol/date range
        """
        print(f"Downloading {self.symbol} data from {self.start_date} to {self.end_date}")
        
        # Download data from Yahoo Finance
        self.data = yf.download(
            self.symbol, 
            start=self.start_date, 
            end=self.end_date,
            progress=False
        )
        
        # Reset index to make Date a column
        self.data = self.data.reset_index()
        
        # Ensure data is not empty
        if self.data.empty:
            raise ValueError(f"Failed to download data for {self.symbol}")
        
        # Save to CSV if path is provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            self.data.to_csv(save_path, index=False)
            print(f"Data saved to {save_path}")
        
        return self.data
    
    def get_technical_indicators(self, df):
        """Calculates common technical indicators for stock analysis
        
        Indicators included:
        - Moving Averages (MA7, MA30)
        - Relative Strength Index (RSI)
        - MACD (Moving Average Convergence Divergence)
        - Bollinger Bands
        
        Parameters:
            df (pd.DataFrame): Stock data with 'Close' column
            
        Returns:
            pd.DataFrame: Stock data with added technical indicators (NaN values removed)
        """
        # Create copy to avoid modifying original data
        df = df.copy()
        
        # Moving Averages (7-day / 30-day)
        df['MA7'] = df['Close'].rolling(window=7).mean()
        df['MA30'] = df['Close'].rolling(window=30).mean()
        
        # Relative Strength Index (RSI, 14-period)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Moving Average Convergence Divergence (MACD)
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands (20-period, 2σ)
        df['Middle_Band'] = df['Close'].rolling(window=20).mean()
        std = df['Close'].rolling(window=20).std()
        df['Upper_Band'] = df['Middle_Band'] + (std * 2)
        df['Lower_Band'] = df['Middle_Band'] - (std * 2)
        
        # Remove rows with NaN values (from rolling calculations)
        df = df.dropna()
        
        return df0