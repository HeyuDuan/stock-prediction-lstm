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
    """LSTM Model for Stock Price Time Series Prediction"""
    
    def __init__(self, input_shape, model_path='models/lstm_model.keras'):
        self.input_shape = input_shape
        self.model_path = model_path
        self.model = None
        self.history = None
        
    def build_model(self, lstm_units=[50, 50], dropout_rate=0.2, learning_rate=0.001):
        """Builds LSTM model architecture for time series prediction
        
        Parameters:
            lstm_units (list): Number of units in each LSTM layer (default: [50, 50])
            dropout_rate (float): Dropout rate for regularization (default: 0.2)
            learning_rate (float): Learning rate for Adam optimizer (default: 0.001)
            
        Returns:
            tf.keras.Model: Compiled LSTM model
        """
        print("Building LSTM model...")
        
        model = Sequential([
            # First LSTM layer (returns sequences for stacked LSTM)
            LSTM(
                units=lstm_units[0],
                return_sequences=True,
                input_shape=self.input_shape,
                kernel_initializer='glorot_uniform',
                recurrent_initializer='orthogonal'
            ),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            # Second LSTM layer (no sequence return for final output)
            LSTM(
                units=lstm_units[1],
                return_sequences=False,
                kernel_initializer='glorot_uniform',
                recurrent_initializer='orthogonal'
            ),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            # Fully connected layer
            Dense(25, activation='relu', kernel_initializer='he_normal'),
            Dense(1, kernel_initializer='glorot_uniform')  # Output layer (price prediction)
        ])
        
        # Compile model with Adam optimizer
        optimizer = Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        model.compile(
            optimizer=optimizer,
            loss='mse',  # Mean Squared Error (regression task)
            metrics=['mae', 'mse']  # Mean Absolute Error, Mean Squared Error
        )
        
        self.model = model
        self._print_model_summary()
        
        return model
    
    def _print_model_summary(self):
        """Prints detailed model architecture and parameter count"""
        print("=" * 60)
        print("Model Architecture Summary")
        print("=" * 60)
        self.model.summary()
        
        # Calculate total parameters
        trainable_params = np.sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
        non_trainable_params = np.sum([tf.keras.backend.count_params(w) for w in self.model.non_trainable_weights])
        
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Non-trainable Parameters: {non_trainable_params:,}")
        print(f"Total Parameters: {trainable_params + non_trainable_params:,}")
        print("=" * 60)
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Trains LSTM model with validation and regularization callbacks
        
        Parameters:
            X_train (np.array): Training input data (shape: [samples, timesteps, features])
            y_train (np.array): Training target values (stock prices)
            X_val (np.array): Validation input data
            y_val (np.array): Validation target values
            epochs (int): Number of training epochs (default: 50)
            batch_size (int): Batch size for training (default: 32)
            
        Returns:
            tf.keras.callbacks.History: Training history object
        """
        print("Starting model training...")
        
        # Create log directory for TensorBoard
        log_dir = f"logs/fit/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Define training callbacks (regularization/early stopping)
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
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print(f"Training completed!")
        print(f"Best model saved to: {self.model_path}")
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """Evaluates model performance on test dataset
        
        Parameters:
            X_test (np.array): Test input data
            y_test (np.array): Test target values
            
        Returns:
            list: Evaluation metrics [loss, mae, mse]
        """
        if self.model is None:
            self.load_model()
        
        print("Evaluating model performance...")
        
        # Calculate evaluation metrics
        evaluation = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Print results
        print("=" * 60)
        print("Model Evaluation Results")
        print("=" * 60)
        print(f"Test Loss (MSE): {evaluation[1]:.6f}")
        print(f"Test MAE: {evaluation[2]:.6f}")
        print("=" * 60)
        
        return evaluation
    
    def predict(self, X):
        """Generates predictions for input time series data
        
        Parameters:
            X (np.array): Input data (shape: [samples, timesteps, features])
            
        Returns:
            np.array: Predicted stock prices
        """
        if self.model is None:
            self.load_model()
        
        return self.model.predict(X, verbose=0)
    
    def save_model(self):
        """Saves trained model to specified path"""
        self.model.save(self.model_path)
        print(f"Model saved to: {self.model_path}")
    
    def load_model(self):
        """Loads pre-trained model from specified path
        
        Returns:
            tf.keras.Model: Loaded LSTM model
        """
        self.model = tf.keras.models.load_model(self.model_path)
        print(f"Model loaded from {self.model_path}")
        return self.model