# src/data_loader.py
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class StockDataLoader:
    def __init__(self, symbol, start_date, end_date):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
    
    def download_data(self, save_path=None):
        """从yfinance下载股票数据"""
        print(f"下载 {self.symbol} 从 {self.start_date} 到 {self.end_date}")
        
        # 下载数据
        self.data = yf.download(
            self.symbol, 
            start=self.start_date, 
            end=self.end_date,
            progress=False
        )
        
        # 重置索引，将日期变为列
        self.data = self.data.reset_index()
        
        # 确保有数据
        if self.data.empty:
            raise ValueError(f"无法下载 {self.symbol} 的数据")
        
        # 保存到CSV
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            self.data.to_csv(save_path, index=False)
            print(f"数据已保存到 {save_path}")
        
        return self.data
    
    def get_technical_indicators(self, df):
        """计算技术指标"""
        # 复制数据
        df = df.copy()
        
        # 移动平均线
        df['MA7'] = df['Close'].rolling(window=7).mean()
        df['MA30'] = df['Close'].rolling(window=30).mean()
        
        # 相对强弱指数 (RSI)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 移动平均收敛发散 (MACD)
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # 布林带
        df['Middle_Band'] = df['Close'].rolling(window=20).mean()
        std = df['Close'].rolling(window=20).std()
        df['Upper_Band'] = df['Middle_Band'] + (std * 2)
        df['Lower_Band'] = df['Middle_Band'] - (std * 2)
        
        # 删除NaN值
        df = df.dropna()
        
        return df0