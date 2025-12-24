# src/data_generator.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class StockDataGenerator:
    """生成模拟股票数据"""
    
    def __init__(self, initial_price=100.0, trend_slope=0.1, volatility=2.0):
        self.initial_price = initial_price
        self.trend_slope = trend_slope
        self.volatility = volatility
        
    def generate_stock_data(self, start_date, end_date, symbol="AAPL"):
        """
        生成模拟股票数据
        
        参数:
            start_date: 开始日期，格式 'YYYY-MM-DD'
            end_date: 结束日期，格式 'YYYY-MM-DD'
            symbol: 股票代码
            
        返回:
            DataFrame: 包含开盘价、最高价、最低价、收盘价、交易量的股票数据
        """
        # 生成日期范围
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        n_days = len(dates)
        
        # 设置随机种子确保可重复性
        np.random.seed(42)
        
        # 生成基本价格序列（带趋势）
        days = np.arange(n_days)
        trend = self.trend_slope * days
        base_prices = self.initial_price + trend
        
        # 添加季节性（年周期）
        seasonal = 10 * np.sin(2 * np.pi * days / 252)  # 252个交易日一年
        
        # 添加随机波动
        random_walk = np.cumsum(np.random.randn(n_days) * self.volatility)
        
        # 组合得到收盘价
        close_prices = base_prices + seasonal + random_walk
        close_prices = np.maximum(close_prices, 1)  # 确保价格为正
        
        # 生成其他价格（开盘、最高、最低）
        open_prices = close_prices * (1 + np.random.normal(0, 0.01, n_days))
        high_prices = close_prices * (1 + np.abs(np.random.normal(0.02, 0.005, n_days)))
        low_prices = close_prices * (1 - np.abs(np.random.normal(0.02, 0.005, n_days)))
        
        # 确保高低价合理
        for i in range(n_days):
            high_prices[i] = max(open_prices[i], close_prices[i], high_prices[i])
            low_prices[i] = min(open_prices[i], close_prices[i], low_prices[i])
        
        # 生成交易量（与价格波动相关）
        price_change = np.abs(np.diff(close_prices, prepend=close_prices[0]))
        volume = np.random.randint(1000000, 10000000, n_days) * (1 + price_change / close_prices)
        
        # 创建DataFrame
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
        """添加技术指标"""
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
        
        return df
    
    def save_to_csv(self, df, filepath):
        """保存数据到CSV文件"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"✅ 数据已保存到: {filepath}")
        print(f"📊 数据形状: {df.shape}")
        print(f"📅 日期范围: {df['Date'].min()} 到 {df['Date'].max()}")