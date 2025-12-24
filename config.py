import sys
import os

# 直接指定项目根目录（根据你的实际路径修改，比如项目在桌面）
project_root = r"C:\Users\86136\Desktop\stock_prediction_lstm"
sys.path.append(project_root)

# 重新验证
print(f"✅ 项目根目录：{project_root}")
print(f"✅ config.py是否存在：{os.path.exists(os.path.join(project_root, 'config.py'))}")
print(f"✅ src/data_generator.py是否存在：{os.path.exists(os.path.join(project_root, 'src', 'data_generator.py'))}")# config.py
class Config:
    """项目配置"""
    
    # 数据生成参数
    STOCK_SYMBOL = "AAPL"  # 模拟苹果股票
    START_DATE = "2020-01-01"
    END_DATE = "2025-12-01"
    INITIAL_PRICE = 100.0  # 初始价格
    TREND_SLOPE = 0.1      # 每日上涨趋势
    VOLATILITY = 2.0       # 波动率
    
    # 模型参数
    LOOKBACK_DAYS = 60     # 用过去60天的数据预测下一天
    TRAIN_SPLIT = 0.8      # 80%训练，20%测试
    FEATURES = ['Close']   # 使用的特征
    
    # 模型训练参数
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    DROPOUT_RATE = 0.2
    
    # 路径配置
    DATA_PATH = "data/stock_data.csv"
    MODEL_PATH = r"C:\Users\86136\Desktop\stock_prediction_lstm\models\lstm_model.keras"
    SCALER_PATH = r"C:\Users\86136\Desktop\stock_prediction_lstm\models\scaler.pkl"
    
    # Web应用配置
    HOST = "localhost"
    PORT = 5000