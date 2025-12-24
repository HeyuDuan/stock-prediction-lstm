# main.py
"""
主训练脚本 - 股票价格预测系统
"""

import sys
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# 设置随机种子保证可复现性
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 添加项目根目录到路径
sys.path.append('.')

from config import Config
from src.data_generator import StockDataGenerator
from src.preprocessor import StockPreprocessor
from src.lstm_model import LSTMModel
from src.visualization import StockVisualizer

def setup_environment():
    """设置运行环境"""
    print("=" * 60)
    print("📈 股票价格预测系统 - LSTM模型训练")
    print("=" * 60)
    
    # 检查TensorFlow版本
    print(f"TensorFlow版本: {tf.__version__}")
    print(f"NumPy版本: {np.__version__}")
    
    # 检查GPU是否可用
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU可用: {[gpu.name for gpu in gpus]}")
    else:
        print("⚠️  GPU不可用，使用CPU")

def generate_or_load_data():
    """生成或加载数据"""
    print("\n" + "=" * 60)
    print("📊 数据准备阶段")
    print("=" * 60)
    
    try:
        # 尝试加载已有数据
        import pandas as pd
        data = pd.read_csv(Config.DATA_PATH)
        print(f"✅ 从 {Config.DATA_PATH} 加载已有数据")
        print(f"   数据形状: {data.shape}")
        
    except FileNotFoundError:
        # 生成新数据
        print("📁 未找到数据文件，生成新的模拟数据...")
        
        generator = StockDataGenerator(
            initial_price=Config.INITIAL_PRICE,
            trend_slope=Config.TREND_SLOPE,
            volatility=Config.VOLATILITY
        )
        
        # 生成基本数据
        data = generator.generate_stock_data(
            start_date=Config.START_DATE,
            end_date=Config.END_DATE,
            symbol=Config.STOCK_SYMBOL
        )
        
        # 添加技术指标
        data = generator.add_technical_indicators(data)
        
        # 保存数据
        generator.save_to_csv(data, Config.DATA_PATH)
    
    return data

def prepare_data():
    """准备训练和测试数据"""
    print("\n" + "=" * 60)
    print("🔧 数据预处理")
    print("=" * 60)
    
    # 初始化预处理器
    preprocessor = StockPreprocessor(
        lookback_days=Config.LOOKBACK_DAYS,
        feature_columns=Config.FEATURES
    )
    
    # 加载并准备数据
    X_train, y_train, X_test, y_test, original_df = preprocessor.load_and_prepare_data(
        Config.DATA_PATH,
        train_split=Config.TRAIN_SPLIT
    )
    
    # 进一步分割训练集为训练和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, 
        test_size=0.1, 
        random_state=SEED
    )
    
    print(f"\n📁 数据集分割结果:")
    print(f"   训练集: {X_train.shape} (用于训练)")
    print(f"   验证集: {X_val.shape} (用于验证)")
    print(f"   测试集: {X_test.shape} (用于最终测试)")
    
    # 保存scaler
    preprocessor.save_scaler(Config.SCALER_PATH)
    
    return X_train, y_train, X_val, y_val, X_test, y_test, preprocessor, original_df

def build_and_train_model(X_train, y_train, X_val, y_val):
    """构建和训练模型"""
    print("\n" + "=" * 60)
    print("🧠 模型构建与训练")
    print("=" * 60)
    
    # 构建模型
    input_shape = (X_train.shape[1], X_train.shape[2])
    lstm_model = LSTMModel(input_shape, Config.MODEL_PATH)
    
    model = lstm_model.build_model(
        lstm_units=[50, 50],
        dropout_rate=Config.DROPOUT_RATE,
        learning_rate=Config.LEARNING_RATE
    )
    
    # 训练模型
    history = lstm_model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE
    )
    
    return lstm_model, history

def evaluate_model(lstm_model, X_test, y_test, preprocessor):
    """评估模型性能"""
    print("\n" + "=" * 60)
    print("📈 模型评估")
    print("=" * 60)
    
    # 评估模型
    evaluation = lstm_model.evaluate(X_test, y_test)
    
    # 进行预测
    predictions_scaled = lstm_model.predict(X_test)
    
    # 反标准化
    predictions = preprocessor.inverse_transform(
        predictions_scaled.reshape(-1, 1)
    ).flatten()
    
    y_test_actual = preprocessor.inverse_transform(
        y_test.reshape(-1, 1)
    ).flatten()
    
    return predictions, y_test_actual, evaluation

def visualize_results(history, y_true, y_pred, original_df):
    """可视化结果"""
    print("\n" + "=" * 60)
    print("🎨 结果可视化")
    print("=" * 60)
    
    visualizer = StockVisualizer()
    
    # 1. 绘制训练历史
    visualizer.plot_training_history(
        history, 
        save_path="static/images/training_history.png"
    )
    
    # 2. 绘制预测结果
    metrics = visualizer.plot_predictions(
        y_true, y_pred,
        save_path="static/images/predictions.png"
    )
    
    # 3. 绘制特征相关性
    visualizer.plot_feature_correlation(
        original_df.select_dtypes(include=[np.number]),
        save_path="static/images/correlation_heatmap.png"
    )
    
    return metrics

def main():
    """主函数"""
    try:
        # 1. 环境设置
        setup_environment()
        
        # 2. 数据准备
        data = generate_or_load_data()
        
        # 3. 数据预处理
        (X_train, y_train, X_val, y_val, 
         X_test, y_test, preprocessor, original_df) = prepare_data()
        
        # 4. 构建和训练模型
        lstm_model, history = build_and_train_model(X_train, y_train, X_val, y_val)
        
        # 5. 评估模型
        predictions, y_true, evaluation = evaluate_model(
            lstm_model, X_test, y_test, preprocessor
        )
        
        # 6. 可视化结果
        metrics = visualize_results(history, y_true, predictions, original_df)
        
        # 7. 输出最终结果
        print("\n" + "=" * 60)
        print("🎉 项目完成!")
        print("=" * 60)
        print(f"\n📁 生成的文件:")
        print(f"   数据文件: {Config.DATA_PATH}")
        print(f"   模型文件: {Config.MODEL_PATH}")
        print(f"   Scaler文件: {Config.SCALER_PATH}")
        print(f"   可视化图表: static/images/")
        
        print(f"\n📊 最终模型性能:")
        for metric, value in metrics.items():
            print(f"   {metric.upper()}: {value:.4f}")
        
        print("\n✅ 项目运行成功！可以启动Web应用查看交互界面。")
        print("   运行命令: cd app && python app.py")
        
    except Exception as e:
        print(f"\n❌ 运行出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())