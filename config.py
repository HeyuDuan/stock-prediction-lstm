import sys
import os

# Specify project root directory (modify to your actual path)
project_root = r"C:\Users\86136\Desktop\stock_prediction_lstm"
sys.path.append(project_root)

# Validate file existence
print(f"Project Root Directory: {project_root}")
print(f"config.py Exists: {os.path.exists(os.path.join(project_root, 'config.py'))}")
print(f"src/data_generator.py Exists: {os.path.exists(os.path.join(project_root, 'src', 'data_generator.py'))}")
class Config:
    """Configuration for LSTM Stock Price Prediction Project"""
    
    # Data Generation Params
    STOCK_SYMBOL = "AAPL"  # Apple stock (simulated)
    START_DATE = "2020-01-01"
    END_DATE = "2025-12-01"
    INITIAL_PRICE = 100.0  # Initial stock price
    TREND_SLOPE = 0.1      # Daily upward trend
    VOLATILITY = 2.0       # Price volatility
    
    # Model Params
    LOOKBACK_DAYS = 60     # Predict next day with 60-day history
    TRAIN_SPLIT = 0.8      # 80% train / 20% test
    FEATURES = ['Close']   # Features for prediction
    
    # Training Params
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    DROPOUT_RATE = 0.2
    
    # Path Config
    DATA_PATH = "data/stock_data.csv"
    MODEL_PATH = r"C:\Users\86136\Desktop\stock_prediction_lstm\models\lstm_model.keras"
    SCALER_PATH = r"C:\Users\86136\Desktop\stock_prediction_lstm\models\scaler.pkl"
    
    # Web App Config
    HOST = "localhost"
    PORT = 5000