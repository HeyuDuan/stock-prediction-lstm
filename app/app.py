# app/app.py
from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import os
import sys
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import Config
from src.data_generator import StockDataGenerator

app = Flask(__name__)

class WebPredictor:
    """Web应用预测器"""
    
    def __init__(self):
        # 加载模型
        self.model = tf.keras.models.load_model(Config.MODEL_PATH)
        
        # 加载scaler
        with open(Config.SCALER_PATH, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # 初始化数据生成器
        self.data_generator = StockDataGenerator(
            initial_price=Config.INITIAL_PRICE,
            trend_slope=Config.TREND_SLOPE,
            volatility=Config.VOLATILITY
        )
        
        self.lookback = Config.LOOKBACK_DAYS
        self.feature_columns = Config.FEATURES
    
    def predict_next_day(self, symbol="AAPL", days_back=100):
        """预测下一天的价格"""
        try:
            # 生成模拟数据
            data = self.data_generator.generate_stock_data(
                start_date=(datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d'),
                end_date=datetime.now().strftime('%Y-%m-%d'),
                symbol=symbol
            )
            
            # 提取特征并标准化
            feature_data = data[self.feature_columns].values[-self.lookback:]
            scaled_data = self.scaler.transform(feature_data)
            
            # 重塑为模型输入格式
            input_data = scaled_data.reshape(1, self.lookback, len(self.feature_columns))
            
            # 进行预测
            prediction_scaled = self.model.predict(input_data, verbose=0)
            
            # 反标准化
            prediction = self.scaler.inverse_transform(prediction_scaled)[0][0]
            
            # 获取最新价格
            current_price = data['Close'].iloc[-1]
            
            # 计算历史价格
            recent_prices = data['Close'].values[-30:]  # 最近30天
            
            return {
                'symbol': symbol,
                'current_price': float(current_price),
                'predicted_price': float(prediction),
                'change': float(prediction - current_price),
                'change_percent': float((prediction - current_price) / current_price * 100),
                'recent_prices': recent_prices.tolist(),
                'recent_dates': data['Date'].dt.strftime('%Y-%m-%d').values[-30:].tolist(),
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }

# 初始化预测器
predictor = WebPredictor()

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """预测API接口"""
    try:
        # 获取请求参数
        data = request.json
        symbol = data.get('symbol', Config.STOCK_SYMBOL)
        
        # 进行预测
        result = predictor.predict_next_day(symbol=symbol)
        
        if result['status'] == 'error':
            return jsonify({'error': result['message']}), 400
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model_info', methods=['GET'])
def api_model_info():
    """获取模型信息"""
    try:
        # 获取模型信息
        model_summary = []
        predictor.model.summary(print_fn=lambda x: model_summary.append(x))
        
        return jsonify({
            'model_name': 'LSTM Stock Predictor',
            'lookback_days': Config.LOOKBACK_DAYS,
            'features': Config.FEATURES,
            'training_date': datetime.fromtimestamp(
                os.path.getctime(Config.MODEL_PATH)
            ).strftime('%Y-%m-%d %H:%M:%S'),
            'model_summary': model_summary
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def api_health():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': predictor.model is not None
    })

if __name__ == '__main__':
    print("=" * 50)
    print("🌐 股票预测Web应用")
    print(f"   访问地址: http://{Config.HOST}:{Config.PORT}")
    print("=" * 50)
    
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=True
    )