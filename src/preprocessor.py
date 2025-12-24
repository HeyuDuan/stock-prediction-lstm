# src/preprocessor.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
import os

class StockPreprocessor:
    """股票数据预处理器"""
    
    def __init__(self, lookback_days=60, feature_columns=['Close']):
        self.lookback_days = lookback_days
        self.feature_columns = feature_columns
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def load_and_prepare_data(self, data_path, train_split=0.8):
        """
        加载并准备数据
        返回: X_train, y_train, X_test, y_test, original_df
        """
        print("📥 加载数据...")
        df = pd.read_csv(data_path, parse_dates=['Date'])
        
        # 确保数据按日期排序
        df = df.sort_values('Date')
        
        # 提取特征
        feature_data = df[self.feature_columns].values
        
        # 标准化数据
        print("🔄 标准化数据...")
        scaled_data = self.scaler.fit_transform(feature_data)
        
        # 创建时间序列序列
        print("🔧 创建时间序列序列...")
        X, y = self._create_sequences(scaled_data)
        
        # 分割数据集
        split_idx = int(len(X) * train_split)
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_test, y_test = X[split_idx:], y[split_idx:]
        
        print(f"✅ 数据准备完成!")
        print(f"  训练集: X={X_train.shape}, y={y_train.shape}")
        print(f"  测试集: X={X_test.shape}, y={y_test.shape}")
        
        return X_train, y_train, X_test, y_test, df
    
    def _create_sequences(self, data):
        """创建LSTM输入序列"""
        X, y = [], []
        
        for i in range(self.lookback_days, len(data)):
            X.append(data[i-self.lookback_days:i])
            y.append(data[i, 0])  # 预测下一个时间步的收盘价
        
        return np.array(X), np.array(y)
    
    def save_scaler(self, filepath):
        """保存scaler"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"✅ Scaler已保存到: {filepath}")
    
    def load_scaler(self, filepath):
        """加载scaler"""
        with open(filepath, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"✅ Scaler已从 {filepath} 加载")
        return self.scaler
    
    def inverse_transform(self, scaled_data):
        """将标准化数据转换回原始范围"""
        return self.scaler.inverse_transform(scaled_data)