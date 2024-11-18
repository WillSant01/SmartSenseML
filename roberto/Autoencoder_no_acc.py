import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import GRU, Dense, Input, Dropout, MultiHeadAttention, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import glob
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
warnings.filterwarnings('ignore')

# Set TensorFlow to be compatible with Python 3.8.10
tf.compat.v1.disable_eager_execution()

class GestureGRUClassifier:
    def __init__(self, window_size=50, verbose=1):
        self.window_size = window_size
        self.scaler = MinMaxScaler()
        self.verbose = verbose
        self.n_classes = 5  # 0=rest, 1=up, 2=down, 3=left, 4=right
        
    def load_and_preprocess_data(self, rest_folder, gesture_folders):
        """Load data from all folders, preserving the natural imbalance."""
        if self.verbose:
            print("\nStarting data loading process...")
            
        all_sequences = []
        all_labels = []
        
        # First load rest data
        if self.verbose:
            print(f"\nProcessing rest data from: {os.path.abspath(rest_folder)}")
        
        rest_files = sorted(glob.glob(os.path.join(rest_folder, "*.csv")))
        if self.verbose:
            print(f"Found {len(rest_files)} rest files")
        
        # Process rest data
        for file in rest_files:
            try:
                if self.verbose:
                    print(f"Reading rest file: {os.path.basename(file)}")
                
                df = pd.read_csv(file)
                data = df[['GyroX', 'GyroY', 'GyroZ', 'label']].values
                
                # Use larger stride for rest data to reduce volume but maintain representation
                stride = 10  # 0.5 seconds stride for rest
                for i in range(0, len(data) - self.window_size + 1, stride):
                    sequence = data[i:i + self.window_size]
                    if len(sequence) == self.window_size:
                        # Verify it's actually rest data
                        if np.all(sequence[:, -1] == 0):  # all labels are 0
                            gyro_sequence = sequence[:, :3]
                            all_sequences.append(gyro_sequence)
                            all_labels.append(0)
                            
            except Exception as e:
                print(f"Error processing rest file {file}: {str(e)}")
                continue
        
        # Process gesture data with smaller stride
        for gesture_name, folder in gesture_folders.items():
            if self.verbose:
                print(f"\nProcessing {gesture_name} gesture data...")
            
            gesture_files = sorted(glob.glob(os.path.join(folder, "*.csv")))
            if self.verbose:
                print(f"Found {len(gesture_files)} files")
            
            for file in gesture_files:
                try:
                    if self.verbose:
                        print(f"Reading gesture file: {os.path.basename(file)}")
                    
                    df = pd.read_csv(file)
                    data = df[['GyroX', 'GyroY', 'GyroZ', 'label']].values
                    
                    # Smaller stride for gesture data to capture more detail
                    stride = 3  # 0.15 seconds stride for gestures
                    for i in range(0, len(data) - self.window_size + 1, stride):
                        sequence = data[i:i + self.window_size]
                        if len(sequence) == self.window_size:
                            # Check if this window contains a gesture
                            window_labels = sequence[:, -1]
                            label_counts = np.bincount(window_labels.astype(int))
                            majority_label = np.argmax(label_counts)
                            
                            # Only keep windows where the gesture is clearly present
                            if majority_label != 0 and label_counts[majority_label] >= self.window_size * 0.8:
                                gyro_sequence = sequence[:, :3]
                                all_sequences.append(gyro_sequence)
                                all_labels.append(majority_label)
                                
                except Exception as e:
                    print(f"Error processing gesture file {file}: {str(e)}")
                    continue
        
        X = np.array(all_sequences)
        y = np.array(all_labels)
        
        if self.verbose:
            print("\nFinal data shapes:")
            print(f"X: {X.shape}")
            print(f"y: {y.shape}")
            print("\nClass distribution:")
            total_samples = len(y)
            for i in range(self.n_classes):
                count = np.sum(y == i)
                print(f"Class {i}: {count} sequences ({count/total_samples*100:.1f}%)")
        
        # Normalize gyro data
        X_reshaped = X.reshape(-1, 3)
        self.scaler.fit(X_reshaped)
        X_normalized = self.scaler.transform(X_reshaped).reshape(X.shape)
        
        return X_normalized, y
        
    def build_model(self):
        """Build GRU model with attention mechanism"""
        inputs = Input(shape=(self.window_size, 3))  # 3 for GyroX, GyroY, GyroZ
        
        # GRU layers
        x = GRU(128, return_sequences=True)(inputs)
        x = LayerNormalization()(x)
        x = Dropout(0.2)(x)
        
        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=4, 
            key_dim=32
        )(x, x)
        x = LayerNormalization()(attention_output + x)  # Skip connection
        
        # Final GRU layer
        x = GRU(64)(x)
        x = LayerNormalization()(x)
        x = Dropout(0.2)(x)
        
        # Classification head
        x = Dense(32, activation='relu')(x)
        outputs = Dense(self.n_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def train_model(self, rest_folder, gesture_folders, epochs=10, batch_size=32):
        """Train the GRU model on imbalanced data."""
        # Load and preprocess data
        X, y = self.load_and_preprocess_data(rest_folder, gesture_folders)
        
        # Split data WITHOUT stratification to maintain natural imbalance
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        shuffle=True
    )
        
        if self.verbose:
            print("\nTraining data distribution:")
            for i in range(self.n_classes):
                count = np.sum(self.y_train == i)
                print(f"Class {i}: {count} sequences ({count/len(self.y_train)*100:.1f}%)")
        
        # Calculate class weights to help with imbalance
        total_samples = len(self.y_train)
        n_samples_per_class = np.bincount(self.y_train.astype(int))
        class_weights = {
            i: total_samples / (self.n_classes * count) 
            for i, count in enumerate(n_samples_per_class)
        }
        
        if self.verbose:
            print("\nClass weights:")
            for class_idx, weight in class_weights.items():
                print(f"Class {class_idx}: {weight:.2f}")
        
        # Build and train model with class weights
        self.model = self.build_model()
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,  # Increased patience due to imbalanced data
            restore_best_weights=True,
            mode='min',
            verbose=self.verbose
        )
        
        self.history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            class_weight=class_weights,  # Apply class weights
            verbose=self.verbose,
            callbacks=[early_stopping]
        )
        
        # Evaluate with metrics suitable for imbalanced data
        if self.verbose:
            print("\nEvaluating model...")
            y_pred = self.model.predict(self.X_val)
            y_pred_classes = np.argmax(y_pred, axis=1)
    
        # Calculate per-class metrics
        print("\nPer-class metrics:")
        for i in range(self.n_classes):
            class_mask = self.y_val == i
            if np.any(class_mask):
                class_acc = np.mean(y_pred_classes[class_mask] == self.y_val[class_mask])
                print(f"Class {i} accuracy: {class_acc:.4f}")
                    
    def plot_training_results(self, output_dir):
        plots_dir = os.path.join(output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
    
        # Plot training history
        plt.figure(figsize=(12, 5))
    
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='Training')
        plt.plot(self.history.history['val_loss'], label='Validation')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
    
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['accuracy'], label='Training')
        plt.plot(self.history.history['val_accuracy'], label='Validation')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
    
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'training_history.png'))
        plt.close()
        
    def save_model(self, output_dir="trained_model"):
        """Save trained model and scaler"""
        os.makedirs(output_dir, exist_ok=True)
        
        self.model.save(os.path.join(output_dir, "gesture_model.h5"))
        joblib.dump(self.scaler, os.path.join(output_dir, "scaler.pkl"))
        
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
    
    try:
        # Create and train model
        classifier = GestureGRUClassifier(window_size=50, verbose=1)
        
        # Separate rest folder and gesture folders
        rest_folder = data_dirs['rest']
        gesture_folders = {k: v for k, v in data_dirs.items() if k != 'rest'}
        
        # Train model
        classifier.train_model(
            rest_folder=rest_folder,
            gesture_folders=gesture_folders,
            epochs=10
        )
        
        # Save trained model and plots
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(script_dir, f"trained_model_{timestamp}")
        classifier.save_model(output_dir)
        
        print(f"\nTraining completed! Model and plots saved in: {output_dir}")
        
    except Exception as e:
        print(f"\nError during training process: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()