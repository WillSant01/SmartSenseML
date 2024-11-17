import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import os
import glob
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
warnings.filterwarnings('ignore')


class GestureAutoencoderTrainer:
    def __init__(self, window_size=50, verbose=1):
        self.window_size = window_size
        self.scaler = MinMaxScaler()
        self.models = {}
        self.thresholds = {}
        self.histories = {}
        self.verbose = verbose
        
    def load_and_preprocess_data(self, rest_folder, gesture_folders):
        """Load rest and gesture data, creating sequences with small stride for better coverage."""
        if self.verbose:
            print("\nStarting data loading process...")
            print(f"Rest folder path: {os.path.abspath(rest_folder)}")
        
        sequences = {'rest': []}
        
        # Load rest data with explicit error checking
        rest_files = sorted(glob.glob(os.path.join(rest_folder, "*.csv")))
        if self.verbose:
            print(f"\nFound {len(rest_files)} rest files:")
            for f in rest_files:
                print(f"  - {os.path.basename(f)}")
        
        rest_data = []
        for file in rest_files:
            try:
                if self.verbose:
                    print(f"\nReading file: {os.path.basename(file)}")
                df = pd.read_csv(file)
                
                required_columns = ['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ']
                data = df[required_columns].values
                if self.verbose:
                    print(f"Data shape: {data.shape}")
                rest_data.append(data)
                
            except Exception as e:
                print(f"Error reading file {file}: {str(e)}")
                continue
        
        rest_data = np.vstack(rest_data)
        if self.verbose:
            print(f"\nTotal rest data shape: {rest_data.shape}")
        
        # Create rest sequences
        stride = 5  # 0.25 seconds stride
        for i in range(0, len(rest_data) - self.window_size + 1, stride):
            sequence = rest_data[i:i + self.window_size]
            if len(sequence) == self.window_size:
                sequences['rest'].append(sequence)
        
        if self.verbose:
            print(f"Created {len(sequences['rest'])} rest sequences")
        
        # Load gesture data
        for gesture_name, folder in gesture_folders.items():
            if self.verbose:
                print(f"\nProcessing {gesture_name} gesture data...")
                print(f"Folder path: {os.path.abspath(folder)}")
            
            sequences[gesture_name] = []
            gesture_files = sorted(glob.glob(os.path.join(folder, "*.csv")))
            
            if self.verbose:
                print(f"Found {len(gesture_files)} files:")
                for f in gesture_files:
                    print(f"  - {os.path.basename(f)}")
            
            for file in gesture_files:
                try:
                    if self.verbose:
                        print(f"\nReading file: {os.path.basename(file)}")
                    df = pd.read_csv(file)
                    data = df[required_columns].values
                    
                    for i in range(0, len(data) - self.window_size + 1, stride):
                        sequence = data[i:i + self.window_size]
                        if len(sequence) == self.window_size:
                            sequences[gesture_name].append(sequence)
                            
                except Exception as e:
                    print(f"Error reading file {file}: {str(e)}")
                    continue
            
            if self.verbose:
                print(f"Created {len(sequences[gesture_name])} {gesture_name} sequences")
        
        # Convert lists to arrays and normalize
        if self.verbose:
            print("\nNormalizing data...")
            
        for key in sequences:
            sequences[key] = np.array(sequences[key])
            if self.verbose:
                print(f"{key}: {len(sequences[key])} sequences, shape: {sequences[key].shape}")
        
        all_data = np.vstack([seqs.reshape(-1, 6) for seqs in sequences.values()])
        self.scaler.fit(all_data)
        
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
        """Train separate models for each gesture type."""
        # Load and preprocess all data
        self.sequences = self.load_and_preprocess_data(rest_folder, gesture_folders)
        
        if self.verbose:
            print(f"\nData distribution:")
            for key, data in self.sequences.items():
                print(f"{key}: {len(data)} sequences")
            
        # Define early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,  # Number of epochs to wait before stopping if loss doesn't improve
            restore_best_weights=True,  # Restore model weights from the epoch with the best value of the monitored quantity
            mode='min',  # Looking for minimum validation loss
            verbose=1 if self.verbose else 0
        )
            
        # Train a model for each gesture type
        for gesture_name in gesture_folders.keys():
            if self.verbose:
                print(f"\nTraining model for {gesture_name} gesture...")
            
            model = self.build_model()
            
            # Combine rest data with current gesture data
            training_data = np.vstack([self.sequences['rest'], self.sequences[gesture_name]])
            
            # Train model
            history = model.fit(
                training_data, training_data,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                shuffle=True,
                verbose=self.verbose,
                callbacks=[early_stopping]  # Add early stopping callback
            )
            
            self.histories[gesture_name] = history
            
            # Calculate reconstruction errors for threshold
            rest_pred = model.predict(self.sequences['rest'], verbose=0)
            rest_errors = np.mean(np.abs(self.sequences['rest'] - rest_pred), axis=(1,2))
            
            gesture_pred = model.predict(self.sequences[gesture_name], verbose=0)
            gesture_errors = np.mean(np.abs(self.sequences[gesture_name] - gesture_pred), axis=(1,2))
            
            # Set threshold
            threshold = np.mean([np.max(rest_errors), np.min(gesture_errors)])
            
            # Store model and threshold
            self.models[gesture_name] = model
            self.thresholds[gesture_name] = threshold
            
            if self.verbose:
                print(f"\nResults for {gesture_name}:")
                print(f"Best validation loss: {min(history.history['val_loss']):.4f}")
                print(f"Training stopped at epoch {len(history.history['loss'])}")
                print(f"Threshold: {threshold:.4f}")
                rest_accuracy = np.mean(rest_errors < threshold)
                gesture_accuracy = np.mean(gesture_errors > threshold)
                print(f"Rest detection accuracy: {rest_accuracy:.2f}")
                print(f"Gesture detection accuracy: {gesture_accuracy:.2f}")

    def plot_training_results(self, output_dir):
        """Plot training history and error distribution for each gesture"""
        if self.verbose:
            print("\nGenerating training visualization plots...")
        
        # Create plots directory
        plots_dir = os.path.join(output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot loss history for each gesture
        plt.figure(figsize=(12, 6))
        for gesture_name, history in self.histories.items():
            plt.plot(history.history['loss'], label=f'{gesture_name} - training')
            plt.plot(history.history['val_loss'], label=f'{gesture_name} - validation')
        plt.title('Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, 'training_history.png'))
        plt.close()
        
        # Plot error distributions
        for gesture_name, model in self.models.items():
            plt.figure(figsize=(10, 6))
            
            # Get predictions and errors for both rest and gesture data
            rest_pred = model.predict(self.sequences['rest'], verbose=0)
            rest_errors = np.mean(np.abs(self.sequences['rest'] - rest_pred), axis=(1,2))
            
            gesture_pred = model.predict(self.sequences[gesture_name], verbose=0)
            gesture_errors = np.mean(np.abs(self.sequences[gesture_name] - gesture_pred), axis=(1,2))
            
            # Plot error distributions
            sns.histplot(rest_errors, label='Rest', alpha=0.5)
            sns.histplot(gesture_errors, label=gesture_name, alpha=0.5)
            plt.axvline(self.thresholds[gesture_name], color='r', linestyle='--', 
                       label='Threshold')
            
            plt.title(f'Error Distribution - {gesture_name}')
            plt.xlabel('Reconstruction Error')
            plt.ylabel('Count')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(plots_dir, f'error_distribution_{gesture_name}.png'))
            plt.close()
            
    def save_models(self, output_dir="trained_models"):
        """Save all trained models and thresholds"""
        if self.verbose:
            print(f"\nSaving models and parameters to {output_dir}/...")
            
        os.makedirs(output_dir, exist_ok=True)
        
        for gesture_name, model in self.models.items():
            model.save(os.path.join(output_dir, f"model_{gesture_name}.h5"))
            
        joblib.dump(self.scaler, os.path.join(output_dir, "scaler.pkl"))
        
        with open(os.path.join(output_dir, "thresholds.json"), 'w') as f:
            json.dump(self.thresholds, f)
        
        # Generate and save visualization plots
        self.plot_training_results(output_dir)

def main():
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"\nScript directory: {script_dir}")
    
    # Define data directories relative to the script location
    data_dirs = {
        'rest': os.path.join(script_dir, 'REST'),
        'up': os.path.join(script_dir, 'UP'),
        'down': os.path.join(script_dir, 'DOWN'),
        'left': os.path.join(script_dir, 'LEFT'),
        'right': os.path.join(script_dir, 'RIGHT')
    }
    
    # Print directory paths
    print("\nData directories:")
    for name, path in data_dirs.items():
        print(f"{name}: {path}")
        print(f"Exists: {os.path.exists(path)}")
        csv_files = glob.glob(os.path.join(path, "*.csv"))
        print(f"CSV files found: {len(csv_files)}")
    
    # Create trainer instance
    trainer = GestureAutoencoderTrainer(window_size=50, verbose=1)
    
    # Train models for each gesture
    gesture_folders = {k: v for k, v in data_dirs.items() if k != 'rest'}
    
    try:
        print("\nStarting training process...")
        
        # Try to read one file to verify data format
        sample_file = glob.glob(os.path.join(data_dirs['rest'], "*.csv"))[0]
        print(f"\nTrying to read sample file: {sample_file}")
        df = pd.read_csv(sample_file)
        print(f"Sample file columns: {df.columns.tolist()}")
        
        trainer.train_models(
            rest_folder=data_dirs['rest'],
            gesture_folders=gesture_folders,
            epochs=50
        )
        
        # Save trained models and parameters
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(script_dir, f"trained_models_{timestamp}")
        trainer.save_models(output_dir)
        
        print("\nTraining completed!")
        print(f"Model files and visualization plots saved in: {output_dir}")
        print("\nThresholds for each gesture:")
        for gesture, threshold in trainer.thresholds.items():
            print(f"{gesture}: {threshold:.4f}")
            
    except Exception as e:
        print(f"\nError during training process: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()