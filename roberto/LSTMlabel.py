import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
import tensorflow as tf
from collections import Counter
from pathlib import Path
import json
import warnings
import matplotlib.pyplot as plt

# Filter warnings
warnings.filterwarnings('ignore', category=UserWarning)

class DataLoader:
    def __init__(self, base_path, use_acc=False, max_rest_samples=None):
        self.base_path = Path(base_path)
        self.features = ['GyroX', 'GyroY', 'GyroZ']
        if use_acc:
            self.features.extend(['AccX', 'AccY', 'AccZ'])
        self.gesture_map = {}
        self.reverse_gesture_map = {}
        self.max_rest_samples = max_rest_samples
        
    def split_into_windows(self, df, min_length=15, max_length=50):
        """Split data into variable-length windows for REST data."""
        windows = []
        current_idx = 0
        
        while current_idx < len(df) - min_length:
            window_length = np.random.randint(min_length, min(max_length + 1, len(df) - current_idx))
            windows.append(df.iloc[current_idx:current_idx + window_length][self.features].values)
            current_idx += window_length
            
        return windows

    def find_labeled_sequences(self, df, label_value):
        
            sequences = []
            current_seq = []
        
            for idx, row in df.iterrows():
                if row['label'] == label_value:
                    current_seq.append(row[self.features].values)
                elif current_seq:
                    if len(current_seq) >= 35:  # Minimum sequence length
                        sequences.append(np.array(current_seq))
                    current_seq = []
                
            # Don't forget last sequence
            if current_seq and len(current_seq) >= 35:
                sequences.append(np.array(current_seq))
            
            return sequences

    def load_gesture_data(self, gesture_folder):
        """Load sequences from a gesture folder."""
        gesture_path = self.base_path / gesture_folder
        all_sequences = []
        
        if not gesture_path.exists():
            print(f"Warning: Folder {gesture_path} does not exist!")
            return all_sequences

        csv_files = list(gesture_path.glob('*.csv'))
        print(f"\nProcessing {gesture_folder} folder:")
        print(f"Found {len(csv_files)} CSV files")
        
        label_value = {'REST': 0, 'UP': 1, 'DOWN': 2, 'LEFT': 3, 'RIGHT': 4}[gesture_folder]
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                if 'label' not in df.columns:
                    print(f"Warning: No label column found in {csv_file}")
                    continue
                
                if gesture_folder == 'REST':
                    rest_sequences = self.split_into_windows(df)
                    all_sequences.extend(rest_sequences)
                else:
                    sequences = self.find_labeled_sequences(df, label_value)
                    all_sequences.extend(sequences)
                    
            except Exception as e:
                print(f"Error processing {csv_file.name}: {str(e)}")
                continue
        
        print(f"Successfully loaded {len(all_sequences)} sequences from {gesture_folder}")
        return all_sequences

    def load_all_data(self):
        """Load and combine all gesture data."""
        X = []
        y = []
        
        # Load gesture data first
        gesture_sequences = {}
        for gesture in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
            sequences = self.load_gesture_data(gesture)
            gesture_sequences[gesture] = sequences
            X.extend(sequences)
            y.extend([{'UP': 1, 'DOWN': 2, 'LEFT': 3, 'RIGHT': 4}[gesture]] * len(sequences))
        
        # Load REST data
        rest_sequences = self.load_gesture_data('REST')
        
        # Apply REST sample limit if specified
        if self.max_rest_samples is not None and len(rest_sequences) > self.max_rest_samples:
            print(f"\nLimiting REST sequences from {len(rest_sequences)} to {self.max_rest_samples}")
            indices = np.random.choice(len(rest_sequences), self.max_rest_samples, replace=False)
            rest_sequences = [rest_sequences[i] for i in indices]
        
        X.extend(rest_sequences)
        y.extend([0] * len(rest_sequences))
        
        # Create gesture map
        unique_labels = sorted(set(y))
        self.gesture_map = {str(label): int(i) for i, label in enumerate(unique_labels)}
        self.reverse_gesture_map = {int(idx): str(label) for label, idx in self.gesture_map.items()}
        
        print(f"\nFinal dataset statistics:")
        print(f"Total sequences: {len(X)}")
        print("Class distribution:", Counter(y))
        print("Gesture mapping:", self.gesture_map)
        
        return X, np.array(y)

class CustomSequence(tf.keras.utils.Sequence):
    """Custom sequence class for batching variable length data."""
    def __init__(self, X, y, batch_size):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.indices = np.arange(len(X))
        np.random.shuffle(self.indices)
        
    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_X = [self.X[i] for i in batch_idx]
        batch_y = self.y[batch_idx]
        
        # Pad to max length in batch
        max_length = max(len(seq) for seq in batch_X)
        padded_X = np.zeros((len(batch_X), max_length, batch_X[0].shape[1]))
        
        for i, seq in enumerate(batch_X):
            padded_X[i, :len(seq)] = seq
            
        return padded_X, batch_y
    
    def on_epoch_end(self):
        np.random.shuffle(self.indices)

def create_model(input_shape, n_classes):
    """Create a simple LSTM model."""
    inputs = Input(shape=(None, input_shape[1]))
    
    x = LSTM(32, return_sequences=True)(inputs)
    x = Dropout(0.3)(x)
    x = LSTM(16)(x)
    x = Dropout(0.3)(x)
    x = Dense(16, activation='relu')(x)
    outputs = Dense(n_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model

def train_model(base_path, use_acc=False, max_rest_samples=None, epochs=20, batch_size=64):
    """Train model with configurable REST samples."""
    # Initialize data loader with REST sample limit
    data_loader = DataLoader(base_path, use_acc=use_acc, max_rest_samples=max_rest_samples)
    X, y = data_loader.load_all_data()
    
    # Split data while maintaining sequence integrity
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )
    
    # Create data sequences
    train_seq = CustomSequence(X_train, y_train, batch_size)
    val_seq = CustomSequence(X_val, y_val, batch_size)
    
    # Create model
    input_shape = (None, len(data_loader.features))
    n_classes = len(data_loader.gesture_map)
    model = create_model(input_shape, n_classes)
    
    # Add learning rate reduction
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.0001
    )
    
    # Train
    history = model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=epochs,
        callbacks=[reduce_lr],
        verbose=1
    )
    
    # Final evaluation
    val_loss, val_acc = model.evaluate(val_seq, verbose=0)
    print(f"\nFinal validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
    
    # Save model and configuration
    save_path = Path(base_path) / 'trained_model'
    save_path.mkdir(exist_ok=True)
    
    try:
        model.save(str(save_path / 'gesture_model.h5'))
        
        config = {
            'features': data_loader.features,
            'gesture_map': {str(k): int(v) for k, v in data_loader.gesture_map.items()},
            'final_accuracy': float(val_acc),
            'final_loss': float(val_loss)
        }
        
        with open(save_path / 'config.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
            
        print(f"\nModel and configuration saved to: {save_path}")
        
    except Exception as e:
        print(f"Warning during model save: {e}")
    
    return model, history, config

def plot_training_history(history):
    """Plot training history."""
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    base_path = r"C:\Users\Deker\Desktop\The Void\Python\SmartSenseML\roberto"
    
    try:
        model, history, config = train_model(
            base_path,
            use_acc=False,         # Whether to use accelerometer data
            max_rest_samples=100,  # Limit REST samples (None for no limit)
            epochs=20,             # Reduced from 50 to 20
            batch_size=64
        )
        
        # Plot training history
        plot_training_history(history)
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise