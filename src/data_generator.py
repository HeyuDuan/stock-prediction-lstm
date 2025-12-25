import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class StockDataGenerator:
    """Generates simulated stock price data with technical indicators"""
    
    def __init__(self, initial_price=100.0, trend_slope=0.1, volatility=2.0):
        self.initial_price = initial_price
        self.trend_slope = trend_slope
        self.volatility = volatility
        
    def generate_stock_data(self, start_date, end_date, symbol="AAPL"):
        """
        Generates simulated daily stock price data
        
        Parameters:
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            symbol (str): Stock ticker symbol (default: "AAPL")
            
        Returns:
            pd.DataFrame: Stock data with Open/High/Low/Close/Volume columns
        """
        # Generate date range
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        n_days = len(dates)
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Generate base price sequence with trend
        days = np.arange(n_days)
        trend = self.trend_slope * days
        base_prices = self.initial_price + trend
        
        # Add seasonal component (annual cycle)
        seasonal = 10 * np.sin(2 * np.pi * days / 252)  # 252 trading days per year
        
        # Add random walk volatility
        random_walk = np.cumsum(np.random.randn(n_days) * self.volatility)
        
        # Combine to get close prices (ensure non-negative)
        close_prices = base_prices + seasonal + random_walk
        close_prices = np.maximum(close_prices, 1)
        
        # Generate Open/High/Low prices
        open_prices = close_prices * (1 + np.random.normal(0, 0.01, n_days))
        high_prices = close_prices * (1 + np.abs(np.random.normal(0.02, 0.005, n_days)))
        low_prices = close_prices * (1 - np.abs(np.random.normal(0.02, 0.005, n_days)))
        
        # Ensure price logic consistency (High >= Open/Close, Low <= Open/Close)
        for i in range(n_days):
            high_prices[i] = max(open_prices[i], close_prices[i], high_prices[i])
            low_prices[i] = min(open_prices[i], close_prices[i], low_prices[i])
        
        # Generate volume (correlated with price volatility)
        price_change = np.abs(np.diff(close_prices, prepend=close_prices[0]))
        volume = np.random.randint(1000000, 10000000, n_days) * (1 + price_change / close_prices)
        
        # Create DataFrame
        data = pd.DataFrame({
            'Date': dates,
            'Symbol': symbol,
            'Open': open_prices,
            'High': high_prices,
            'Low': low_prices,
            'Close': close_prices,
            'Volume': volume.astype(int)
        })
        
        return data
    
    def add_technical_indicators(self, df):
        """Adds common technical indicators to stock data
        
        Indicators included:
        - Moving Averages (MA7, MA30)
        - Relative Strength Index (RSI)
        - MACD (Moving Average Convergence Divergence)
        - Bollinger Bands
        """
        df = df.copy()
        
        # Moving Averages
        df['MA7'] = df['Close'].rolling(window=7).mean()
        df['MA30'] = df['Close'].rolling(window=30).mean()
        
        # Relative Strength Index (RSI)
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
        
        # Bollinger Bands
        df['Middle_Band'] = df['Close'].rolling(window=20).mean()
        std = df['Close'].rolling(window=20).std()
        df['Upper_Band'] = df['Middle_Band'] + (std * 2)
        df['Lower_Band'] = df['Middle_Band'] - (std * 2)
        
        return df
    
    def save_to_csv(self, df, filepath):
        """Saves stock data DataFrame to CSV file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"Data saved to: {filepath}")
        print(f"Data shape: {df.shape}")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")