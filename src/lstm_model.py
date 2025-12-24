# src/lstm_model.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, 
    ReduceLROnPlateau, 
    ModelCheckpoint,
    TensorBoard
)
import matplotlib.pyplot as plt
import os
from datetime import datetime

class LSTMModel:
    """LSTM股票预测模型"""
    
    def __init__(self, input_shape, model_path='models/lstm_model.keras'):
        self.input_shape = input_shape
        self.model_path = model_path
        self.model = None
        self.history = None
        
    def build_model(self, lstm_units=[50, 50], dropout_rate=0.2, learning_rate=0.001):
        """构建LSTM模型架构"""
        print("🏗️ 构建LSTM模型...")
        
        model = Sequential([
            # 第一层LSTM
            LSTM(
                units=lstm_units[0],
                return_sequences=True,
                input_shape=self.input_shape,
                kernel_initializer='glorot_uniform',
                recurrent_initializer='orthogonal'
            ),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            # 第二层LSTM
            LSTM(
                units=lstm_units[1],
                return_sequences=False,
                kernel_initializer='glorot_uniform',
                recurrent_initializer='orthogonal'
            ),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            # 全连接层
            Dense(25, activation='relu', kernel_initializer='he_normal'),
            Dense(1, kernel_initializer='glorot_uniform')  # 输出层
        ])
        
        # 编译模型
        optimizer = Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        model.compile(
            optimizer=optimizer,
            loss='mse',  # 均方误差
            metrics=['mae', 'mse']  # 平均绝对误差和均方误差
        )
        
        self.model = model
        self._print_model_summary()
        
        return model
    
    def _print_model_summary(self):
        """打印模型摘要"""
        print("=" * 60)
        print("📊 模型架构摘要")
        print("=" * 60)
        self.model.summary()
        
        # 计算总参数
        trainable_params = np.sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
        non_trainable_params = np.sum([tf.keras.backend.count_params(w) for w in self.model.non_trainable_weights])
        
        print(f"可训练参数: {trainable_params:,}")
        print(f"不可训练参数: {non_trainable_params:,}")
        print(f"总参数: {trainable_params + non_trainable_params:,}")
        print("=" * 60)
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """训练模型"""
        print("🚀 开始训练模型...")
        
        # 创建日志目录
        log_dir = f"logs/fit/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # 定义回调函数
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=self.model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            TensorBoard(log_dir=log_dir, histogram_freq=1)
        ]
        
        # 训练模型
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print(f"✅ 训练完成!")
        print(f"📁 最佳模型已保存到: {self.model_path}")
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """评估模型性能"""
        if self.model is None:
            self.load_model()
        
        print("📈 评估模型性能...")
        
        # 计算评估指标
        evaluation = self.model.evaluate(X_test, y_test, verbose=0)
        
        # 打印结果
        print("=" * 60)
        print("📊 模型评估结果")
        print("=" * 60)
        print(f"测试损失 (MSE): {evaluation[1]:.6f}")
        print(f"测试MAE: {evaluation[2]:.6f}")
        print("=" * 60)
        
        return evaluation
    
    def predict(self, X):
        """进行预测"""
        if self.model is None:
            self.load_model()
        
        return self.model.predict(X, verbose=0)
    
    def save_model(self):
        """保存模型"""
        self.model.save(self.model_path)
        print(f"✅ 模型已保存到: {self.model_path}")
    
    def load_model(self):
        """加载模型"""
        self.model = tf.keras.models.load_model(self.model_path)
        print(f"✅ 模型已从 {self.model_path} 加载")
        return self.model