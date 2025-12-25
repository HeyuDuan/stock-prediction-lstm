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
    """Visualization Tool for Stock Price Prediction (LSTM Model)"""
    
    def __init__(self, figsize=(15, 10)):
        self.figsize = figsize
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def plot_training_history(self, history, save_path=None):
        """Plots model training history (loss, MAE, learning rate)
        
        Parameters:
            history (tf.keras.callbacks.History): Training history object from model.fit()
            save_path (str, optional): Path to save plot (default: None)
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. Loss function (MSE)
        axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
        axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].set_title('Model Loss (MSE)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epochs', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. MAE (Mean Absolute Error)
        axes[1].plot(history.history['mae'], label='Training MAE', linewidth=2)
        axes[1].plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
        axes[1].set_title('Mean Absolute Error (MAE)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epochs', fontsize=12)
        axes[1].set_ylabel('MAE', fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. Learning rate evolution
        if 'lr' in history.history:
            axes[2].plot(history.history['lr'], linewidth=2, color='purple')
            axes[2].set_title('Learning Rate Evolution', fontsize=14, fontweight='bold')
            axes[2].set_xlabel('Epochs', fontsize=12)
            axes[2].set_ylabel('Learning Rate', fontsize=12)
            axes[2].set_yscale('log')
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to: {save_path}")
        
        plt.show()
    
    def plot_predictions(self, y_true, y_pred, dates=None, save_path=None):
        """Plots comparison of predicted vs actual stock prices with error analysis
        
        Parameters:
            y_true (np.array): Actual stock prices (test set)
            y_pred (np.array): Predicted stock prices from LSTM model
            dates (np.array, optional): Date labels for x-axis (default: None)
            save_path (str, optional): Path to save plot (default: None)
            
        Returns:
            dict: Performance metrics (MSE, MAE, RMSE, R², MAPE)
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Prediction vs Actual comparison
        axes[0, 0].plot(y_true, label='Actual Values', linewidth=2, alpha=0.8)
        axes[0, 0].plot(y_pred, label='Predicted Values', linewidth=2, alpha=0.8, linestyle='--')
        axes[0, 0].set_title('Stock Price Prediction Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Time Steps', fontsize=12)
        axes[0, 0].set_ylabel('Price (USD)', fontsize=12)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Prediction error distribution
        errors = y_true - y_pred
        axes[0, 1].hist(errors, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        axes[0, 1].axvline(x=errors.mean(), color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: {errors.mean():.3f}')
        axes[0, 1].set_title('Prediction Error Distribution', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Error', fontsize=12)
        axes[0, 1].set_ylabel('Frequency', fontsize=12)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Actual vs Predicted scatter plot
        axes[1, 0].scatter(y_true, y_pred, alpha=0.6, s=20)
        
        # Add perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[1, 0].plot([min_val, max_val], [min_val, max_val], 
                       'r--', linewidth=2, label='Perfect Prediction')
        
        axes[1, 0].set_title('Actual vs Predicted Values', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Actual Price (USD)', fontsize=12)
        axes[1, 0].set_ylabel('Predicted Price (USD)', fontsize=12)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Cumulative absolute error
        cumulative_error = np.cumsum(np.abs(errors))
        axes[1, 1].plot(cumulative_error, linewidth=2, color='green')
        axes[1, 1].set_title('Cumulative Absolute Error', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Time Steps', fontsize=12)
        axes[1, 1].set_ylabel('Cumulative Error', fontsize=12)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add performance metrics text
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
            print(f"Prediction results plot saved to: {save_path}")
        
        plt.show()
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }
    
    def plot_feature_correlation(self, df, save_path=None):
        """Plots correlation heatmap of numerical stock features
        
        Parameters:
            df (pd.DataFrame): Stock data with numerical features
            save_path (str, optional): Path to save plot (default: None)
        """
        # Calculate correlation matrix
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_cols].corr()
        
        plt.figure(figsize=(12, 10))
        
        # Create heatmap (upper triangle masked)
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
        
        plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Correlation heatmap saved to: {save_path}")
        
        plt.show()