import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler
import os
import glob
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

class GestureAutoencoderTrainer:
    def __init__(self, window_size=50):
        self.window_size = window_size
        self.scaler = MinMaxScaler()
        self.models = {}
        self.thresholds = {}
        
    def load_and_preprocess_data(self, rest_folder, gesture_folders):
        """
        Load rest and gesture data, creating sequences with small stride for better coverage.
        
        Args:
            rest_folder: Path to folder containing rest CSVs
            gesture_folders: Dict of gesture_name: folder_path
        Returns:
            Dictionary containing sequences for rest and each gesture type
        """
        print("Loading and preprocessing data...")
        sequences = {'rest': []}
        
        # Load rest data
        rest_files = glob.glob(os.path.join(rest_folder, "*.csv"))
        rest_data = []
        for file in rest_files:
            df = pd.read_csv(file)
            data = df[['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ']].values
            rest_data.append(data)
        rest_data = np.vstack(rest_data)
        
        # Create rest sequences with small stride (5 samples = 0.25s) for better coverage
        stride = 5  # 0.25 seconds stride
        for i in range(0, len(rest_data) - self.window_size + 1, stride):
            sequence = rest_data[i:i + self.window_size]
            if len(sequence) == self.window_size:
                sequences['rest'].append(sequence)
        
        # Load gesture data
        for gesture_name, folder in gesture_folders.items():
            sequences[gesture_name] = []
            gesture_files = glob.glob(os.path.join(folder, "*.csv"))
            
            for file in gesture_files:
                df = pd.read_csv(file)
                data = df[['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ']].values
                
                # Use small stride for gestures too
                for i in range(0, len(data) - self.window_size + 1, stride):
                    sequence = data[i:i + self.window_size]
                    if len(sequence) == self.window_size:
                        sequences[gesture_name].append(sequence)
        
        # Convert lists to arrays
        for key in sequences:
            sequences[key] = np.array(sequences[key])
            
        # Fit scaler on all data
        all_data = np.vstack([seqs.reshape(-1, 6) for seqs in sequences.values()])
        self.scaler.fit(all_data)
        
        # Normalize all sequences
        for key in sequences:
            shape = sequences[key].shape
            sequences[key] = self.scaler.transform(
                sequences[key].reshape(-1, 6)
            ).reshape(shape)
            
        return sequences
        
    def build_model(self):
        """Build autoencoder model"""
        inputs = Input(shape=(self.window_size, 6))
        
        # Encoder
        x = LSTM(128, return_sequences=True)(inputs)
        x = Dropout(0.2)(x)
        x = LSTM(64, return_sequences=True)(x)
        x = Dropout(0.2)(x)
        x = LSTM(32)(x)
        
        # Decoder
        x = Dense(64)(x)
        x = Dense(self.window_size * 6)(x)
        decoded = tf.reshape(x, (-1, self.window_size, 6))
        
        model = Model(inputs, decoded)
        model.compile(optimizer='adam', loss='mse')
        
        return model
        
    def train_models(self, rest_folder, gesture_folders, epochs=50, batch_size=32):
        """
        Train separate models for each gesture type.
        """
        # Load and preprocess all data
        sequences = self.load_and_preprocess_data(rest_folder, gesture_folders)
        
        print(f"\nData distribution:")
        for key, data in sequences.items():
            print(f"{key}: {len(data)} sequences")
            
        # Train a model for each gesture type
        for gesture_name in gesture_folders.keys():
            print(f"\nTraining model for {gesture_name} gesture...")
            
            # Build new model for this gesture
            model = self.build_model()
            
            # Combine rest data with current gesture data
            training_data = np.vstack([sequences['rest'], sequences[gesture_name]])
            
            # Create labels (0 for rest, 1 for gesture)
            labels = np.concatenate([
                np.zeros(len(sequences['rest'])),
                np.ones(len(sequences[gesture_name]))
            ])
            
            # Train model
            history = model.fit(
                training_data, training_data,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                shuffle=True
            )
            
            # Calculate reconstruction errors for threshold
            rest_pred = model.predict(sequences['rest'])
            rest_errors = np.mean(np.abs(sequences['rest'] - rest_pred), axis=(1,2))
            
            gesture_pred = model.predict(sequences[gesture_name])
            gesture_errors = np.mean(np.abs(sequences[gesture_name] - gesture_pred), axis=(1,2))
            
            # Set threshold at point that best separates rest and gesture
            threshold = np.mean([np.max(rest_errors), np.min(gesture_errors)])
            
            # Store model and threshold
            self.models[gesture_name] = model
            self.thresholds[gesture_name] = threshold
            
            print(f"Threshold for {gesture_name}: {threshold}")
            
            # Calculate and print accuracy metrics
            rest_accuracy = np.mean(rest_errors < threshold)
            gesture_accuracy = np.mean(gesture_errors > threshold)
            print(f"Rest detection accuracy: {rest_accuracy:.2f}")
            print(f"Gesture detection accuracy: {gesture_accuracy:.2f}")
            
    def save_models(self, output_dir="trained_models"):
        """Save all trained models and thresholds"""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save models
        for gesture_name, model in self.models.items():
            model.save(os.path.join(output_dir, f"model_{gesture_name}.h5"))
            
        # Save scaler
        import joblib
        joblib.dump(self.scaler, os.path.join(output_dir, "scaler.pkl"))
        
        # Save thresholds
        import json
        with open(os.path.join(output_dir, "thresholds.json"), 'w') as f:
            json.dump(self.thresholds, f)
            
        print(f"\nModels, scaler, and thresholds saved to {output_dir}/")
        
def main():
    # Define data directories
    data_dirs = {
        'rest': 'REST',
        'up': 'UP',
        'down': 'DOWN',
        'left': 'LEFT',
        'right': 'RIGHT',
        'ok': 'OK'
    }
    
    # Create trainer instance
    trainer = GestureAutoencoderTrainer(window_size=50)
    
    # Train models for each gesture
    gesture_folders = {k: v for k, v in data_dirs.items() if k != 'rest'}
    
    print("Starting training process...")
    trainer.train_models(
        rest_folder=data_dirs['rest'],
        gesture_folders=gesture_folders,
        epochs=50
    )
    
    # Save trained models and parameters
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"trained_models_{timestamp}"
    trainer.save_models(output_dir)
    
    print("\nTraining completed!")
    print(f"Model files saved in: {output_dir}")
    print("\nThresholds for each gesture:")
    for gesture, threshold in trainer.thresholds.items():
        print(f"{gesture}: {threshold}")

if __name__ == "__main__":
    main()