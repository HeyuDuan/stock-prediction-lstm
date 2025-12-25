"""
Main Training Script - Stock Price Prediction System
"""

import sys
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Add project root directory to path
sys.path.append('.')

from config import Config
from src.data_generator import StockDataGenerator
from src.preprocessor import StockPreprocessor
from src.lstm_model import LSTMModel
from src.visualization import StockVisualizer

def setup_environment():
    """Set up runtime environment"""
    print("=" * 60)
    print("Stock Price Prediction System - LSTM Model Training")
    print("=" * 60)
    
    # Check TensorFlow version
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"NumPy Version: {np.__version__}")
    
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU Available: {[gpu.name for gpu in gpus]}")
    else:
        print("GPU Not Available, Using CPU")

def generate_or_load_data():
    """Generate or load dataset"""
    print("\n" + "=" * 60)
    print("Data Preparation Phase")
    print("=" * 60)
    
    try:
        # Try to load existing data
        import pandas as pd
        data = pd.read_csv(Config.DATA_PATH)
        print(f"Loaded existing data from {Config.DATA_PATH}")
        print(f"   Data Shape: {data.shape}")
        
    except FileNotFoundError:
        # Generate new data
        print("Data file not found, generating new simulated data...")
        
        generator = StockDataGenerator(
            initial_price=Config.INITIAL_PRICE,
            trend_slope=Config.TREND_SLOPE,
            volatility=Config.VOLATILITY
        )
        
        # Generate base data
        data = generator.generate_stock_data(
            start_date=Config.START_DATE,
            end_date=Config.END_DATE,
            symbol=Config.STOCK_SYMBOL
        )
        
        # Add technical indicators
        data = generator.add_technical_indicators(data)
        
        # Save data
        generator.save_to_csv(data, Config.DATA_PATH)
    
    return data

def prepare_data():
    """Prepare training and testing data"""
    print("\n" + "=" * 60)
    print("Data Preprocessing")
    print("=" * 60)
    
    # Initialize preprocessor
    preprocessor = StockPreprocessor(
        lookback_days=Config.LOOKBACK_DAYS,
        feature_columns=Config.FEATURES
    )
    
    # Load and prepare data
    X_train, y_train, X_test, y_test, original_df = preprocessor.load_and_prepare_data(
        Config.DATA_PATH,
        train_split=Config.TRAIN_SPLIT
    )
    
    # Further split training set into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, 
        test_size=0.1, 
        random_state=SEED
    )
    
    print(f"\nDataset Split Results:")
    print(f"   Training Set: {X_train.shape} (for model training)")
    print(f"   Validation Set: {X_val.shape} (for model validation)")
    print(f"   Test Set: {X_test.shape} (for final testing)")
    
    # Save scaler
    preprocessor.save_scaler(Config.SCALER_PATH)
    
    return X_train, y_train, X_val, y_val, X_test, y_test, preprocessor, original_df

def build_and_train_model(X_train, y_train, X_val, y_val):
    """Build and train LSTM model"""
    print("\n" + "=" * 60)
    print("Model Construction & Training")
    print("=" * 60)
    
    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])
    lstm_model = LSTMModel(input_shape, Config.MODEL_PATH)
    
    model = lstm_model.build_model(
        lstm_units=[50, 50],
        dropout_rate=Config.DROPOUT_RATE,
        learning_rate=Config.LEARNING_RATE
    )
    
    # Train model
    history = lstm_model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE
    )
    
    return lstm_model, history

def evaluate_model(lstm_model, X_test, y_test, preprocessor):
    """Evaluate model performance"""
    print("\n" + "=" * 60)
    print("Model Evaluation")
    print("=" * 60)
    
    # Evaluate model
    evaluation = lstm_model.evaluate(X_test, y_test)
    
    # Make predictions
    predictions_scaled = lstm_model.predict(X_test)
    
    # Inverse transform to original scale
    predictions = preprocessor.inverse_transform(
        predictions_scaled.reshape(-1, 1)
    ).flatten()
    
    y_test_actual = preprocessor.inverse_transform(
        y_test.reshape(-1, 1)
    ).flatten()
    
    return predictions, y_test_actual, evaluation

def visualize_results(history, y_true, y_pred, original_df):
    """Visualize prediction results"""
    print("\n" + "=" * 60)
    print("Result Visualization")
    print("=" * 60)
    
    visualizer = StockVisualizer()
    
    # 1. Plot training history
    visualizer.plot_training_history(
        history, 
        save_path="static/images/training_history.png"
    )
    
    # 2. Plot prediction results
    metrics = visualizer.plot_predictions(
        y_true, y_pred,
        save_path="static/images/predictions.png"
    )
    
    # 3. Plot feature correlation
    visualizer.plot_feature_correlation(
        original_df.select_dtypes(include=[np.number]),
        save_path="static/images/correlation_heatmap.png"
    )
    
    return metrics

def main():
    """Main function"""
    try:
        # 1. Environment setup
        setup_environment()
        
        # 2. Data preparation
        data = generate_or_load_data()
        
        # 3. Data preprocessing
        (X_train, y_train, X_val, y_val, 
         X_test, y_test, preprocessor, original_df) = prepare_data()
        
        # 4. Model construction and training
        lstm_model, history = build_and_train_model(X_train, y_train, X_val, y_val)
        
        # 5. Model evaluation
        predictions, y_true, evaluation = evaluate_model(
            lstm_model, X_test, y_test, preprocessor
        )
        
        # 6. Result visualization
        metrics = visualize_results(history, y_true, predictions, original_df)
        
        # 7. Output final results
        print("\n" + "=" * 60)
        print("Project Completed!")
        print("=" * 60)
        print(f"\nGenerated Files:")
        print(f"   Data File: {Config.DATA_PATH}")
        print(f"   Model File: {Config.MODEL_PATH}")
        print(f"   Scaler File: {Config.SCALER_PATH}")
        print(f"   Visualization Plots: static/images/")
        
        print(f"\nFinal Model Performance:")
        for metric, value in metrics.items():
            print(f"   {metric.upper()}: {value:.4f}")
        
        print("\nProject ran successfully! You can start the web application to view the interactive interface.")
        print("   Run Command: cd app && python app.py")
        
    except Exception as e:
        print(f"\nRuntime Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())