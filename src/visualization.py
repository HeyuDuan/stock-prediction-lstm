# src/visualization.py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score,
    mean_absolute_percentage_error
)
import pandas as pd

class StockVisualizer:
    """股票预测可视化工具"""
    
    def __init__(self, figsize=(15, 10)):
        self.figsize = figsize
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def plot_training_history(self, history, save_path=None):
        """绘制训练历史"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. 损失函数
        axes[0].plot(history.history['loss'], label='训练损失', linewidth=2)
        axes[0].plot(history.history['val_loss'], label='验证损失', linewidth=2)
        axes[0].set_title('模型损失 (MSE)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('训练轮次', fontsize=12)
        axes[0].set_ylabel('损失', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. MAE
        axes[1].plot(history.history['mae'], label='训练MAE', linewidth=2)
        axes[1].plot(history.history['val_mae'], label='验证MAE', linewidth=2)
        axes[1].set_title('平均绝对误差 (MAE)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('训练轮次', fontsize=12)
        axes[1].set_ylabel('MAE', fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. 学习率变化
        if 'lr' in history.history:
            axes[2].plot(history.history['lr'], linewidth=2, color='purple')
            axes[2].set_title('学习率变化', fontsize=14, fontweight='bold')
            axes[2].set_xlabel('训练轮次', fontsize=12)
            axes[2].set_ylabel('学习率', fontsize=12)
            axes[2].set_yscale('log')
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📈 训练历史图已保存到: {save_path}")
        
        plt.show()
    
    def plot_predictions(self, y_true, y_pred, dates=None, save_path=None):
        """绘制预测结果对比"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 预测对比
        axes[0, 0].plot(y_true, label='真实值', linewidth=2, alpha=0.8)
        axes[0, 0].plot(y_pred, label='预测值', linewidth=2, alpha=0.8, linestyle='--')
        axes[0, 0].set_title('股票价格预测对比', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('时间步长', fontsize=12)
        axes[0, 0].set_ylabel('价格', fontsize=12)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 误差分布
        errors = y_true - y_pred
        axes[0, 1].hist(errors, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        axes[0, 1].axvline(x=errors.mean(), color='red', linestyle='--', 
                          linewidth=2, label=f'均值: {errors.mean():.3f}')
        axes[0, 1].set_title('预测误差分布', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('误差', fontsize=12)
        axes[0, 1].set_ylabel('频数', fontsize=12)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 散点图
        axes[1, 0].scatter(y_true, y_pred, alpha=0.6, s=20)
        
        # 添加完美预测线
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[1, 0].plot([min_val, max_val], [min_val, max_val], 
                       'r--', linewidth=2, label='完美预测')
        
        axes[1, 0].set_title('真实值 vs 预测值', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('真实价格', fontsize=12)
        axes[1, 0].set_ylabel('预测价格', fontsize=12)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 累计绝对误差
        cumulative_error = np.cumsum(np.abs(errors))
        axes[1, 1].plot(cumulative_error, linewidth=2, color='green')
        axes[1, 1].set_title('累计绝对误差', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('时间步长', fontsize=12)
        axes[1, 1].set_ylabel('累计误差', fontsize=12)
        axes[1, 1].grid(True, alpha=0.3)
        
        # 添加统计信息
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        
        stats_text = (f'MSE: {mse:.4f}\n'
                     f'MAE: {mae:.4f}\n'
                     f'RMSE: {rmse:.4f}\n'
                     f'R²: {r2:.4f}\n'
                     f'MAPE: {mape:.2f}%')
        
        plt.figtext(0.15, 0.02, stats_text, 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                   fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📈 预测结果图已保存到: {save_path}")
        
        plt.show()
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }
    
    def plot_feature_correlation(self, df, save_path=None):
        """绘制特征相关性热图"""
        # 计算相关性矩阵
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_cols].corr()
        
        plt.figure(figsize=(12, 10))
        
        # 创建热图
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, 
                   mask=mask,
                   annot=True, 
                   fmt='.2f', 
                   cmap='coolwarm',
                   center=0,
                   square=True,
                   linewidths=0.5,
                   cbar_kws={"shrink": 0.8})
        
        plt.title('特征相关性热图', fontsize=16, fontweight='bold', pad=20)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📈 相关性热图已保存到: {save_path}")
        
        plt.show()