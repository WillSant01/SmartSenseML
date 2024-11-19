import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.layers import Input, GRU, Dense, Masking, Layer
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import Counter
import os
from pathlib import Path
import json

class DataLoader:
    def __init__(self, base_path, use_acc=False, rest_samples=20):
        self.base_path = Path(base_path)
        self.use_acc = use_acc
        self.rest_samples = rest_samples
        self.sampling_rate = 20     # 20 Hz
        # Adjust for actual gesture length (50 samples max)
        self.expected_samples = 50  # Max gesture length
        self.features = ['GyroX', 'GyroY', 'GyroZ']
        if use_acc:
            self.features.extend(['AccX', 'AccY', 'AccZ'])
        
        self.gesture_map = {
            'UP': 0,
            'DOWN': 1,
            'LEFT': 2,
            'RIGHT': 3,
            'REST': 4
        }

    def load_gesture_data(self, gesture_folder):
        gesture_path = self.base_path / gesture_folder
        all_sequences = []
        
        if not gesture_path.exists():
            print(f"Warning: Folder {gesture_path} does not exist!")
            return all_sequences, self.gesture_map[gesture_folder]

        csv_files = list(gesture_path.glob('*.csv'))
        print(f"\nProcessing {gesture_folder} folder:")
        print(f"Found {len(csv_files)} CSV files")
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                if gesture_folder == 'REST':
                    # For REST, take segments of expected_samples length
                    n_chunks = len(df) // self.expected_samples
                    for i in range(n_chunks):
                        start_idx = i * self.expected_samples
                        end_idx = start_idx + self.expected_samples
                        chunk = df.iloc[start_idx:end_idx]
                        if len(chunk) == self.expected_samples:
                            all_sequences.append(chunk[self.features].values)
                else:
                    # For gestures, find actual gesture segments
                    # Assuming gestures are separated by non-movement periods
                    # You might need to adjust this threshold based on your data
                    motion_threshold = 0.1
                    
                    # Calculate total motion
                    total_motion = np.sqrt(
                        df['GyroX'].values**2 + 
                        df['GyroY'].values**2 + 
                        df['GyroZ'].values**2
                    )
                    
                    # Find segments with motion
                    in_motion = total_motion > motion_threshold
                    motion_starts = np.where(np.diff(in_motion.astype(int)) == 1)[0]
                    motion_ends = np.where(np.diff(in_motion.astype(int)) == -1)[0]
                    
                    # Process each motion segment
                    for start, end in zip(motion_starts, motion_ends):
                        if end - start >= 35:  # Minimum gesture length
                            segment = df.iloc[start:min(end, start + self.expected_samples)]
                            # Pad if necessary
                            if len(segment) < self.expected_samples:
                                padding_length = self.expected_samples - len(segment)
                                padding = pd.DataFrame(0, index=range(padding_length), columns=self.features)
                                segment = pd.concat([segment[self.features], padding])
                            all_sequences.append(segment[self.features].values)
                
            except Exception as e:
                print(f"Error processing {csv_file.name}: {str(e)}")
                continue
        
        print(f"Successfully loaded {len(all_sequences)} sequences from {gesture_folder}")
        return all_sequences, self.gesture_map[gesture_folder]
    def load_all_data(self):
        """Load and combine all gesture data."""
        X = []
        y = []
    
        # Load gesture data
        for gesture in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
            print(f"\nLoading {gesture} data...")
            sequences, label = self.load_gesture_data(gesture)
            X.extend(sequences)
            y.extend([label] * len(sequences))
            print(f"Total {gesture} sequences: {len(sequences)}")
    
        # Load and sample REST data
        print("\nLoading REST data...")
        rest_sequences, rest_label = self.load_gesture_data('REST')
        if len(rest_sequences) > self.rest_samples:
            indices = np.random.choice(len(rest_sequences), self.rest_samples, replace=False)
            rest_sequences = [rest_sequences[i] for i in indices]
        X.extend(rest_sequences)
        y.extend([rest_label] * len(rest_sequences))
        print(f"Total REST sequences: {len(rest_sequences)}")
    
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
    
        print(f"\nFinal dataset shape: X: {X.shape}, y: {y.shape}")
        print("Class distribution:", Counter(y))
    
        return X, y

class AttentionWithMasking(Layer):
    def __init__(self, **kwargs):
        super(AttentionWithMasking, self).__init__(**kwargs)
        self.score_dense = Dense(1, activation='tanh')

    def call(self, inputs, mask=None):
        hidden_states = inputs
        score = self.score_dense(hidden_states)
        score = tf.squeeze(score, axis=-1)

        if mask is not None:
            mask = tf.cast(mask, dtype=score.dtype)
            score += (1.0 - mask) * -1e9

        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = tf.reduce_sum(hidden_states * tf.expand_dims(attention_weights, -1), axis=1)

        return context_vector, attention_weights

    def compute_mask(self, inputs, mask=None):
        return None

def create_model(input_shape, n_classes=5):
    inputs = Input(shape=input_shape)
    masked_inputs = Masking(mask_value=0.0)(inputs)
    
    # Add dropout to prevent overfitting
    x = tf.keras.layers.Dropout(0.2)(masked_inputs)
    
    # Reduce GRU size and add regularization
    gru_output = GRU(32, return_sequences=True, 
                     kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    
    attention_layer = AttentionWithMasking()
    context_vector, _ = attention_layer(gru_output)
    
    # Add dropout before final layers
    x = tf.keras.layers.Dropout(0.2)(context_vector)
    
    # Reduce dense layer size
    dense_output = Dense(16, activation='relu')(x)
    outputs = Dense(n_classes, activation='softmax')(dense_output)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Use a lower learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(optimizer=optimizer, 
                 loss='categorical_crossentropy', 
                 metrics=['accuracy'])
    
    return model

def train_model(base_path, use_acc=False, rest_samples=20, epochs=50, batch_size=32):
    # Initialize data loader
    data_loader = DataLoader(base_path, use_acc=use_acc, rest_samples=rest_samples)
    
    # Load and preprocess data
    X, y = data_loader.load_all_data()
    
    # Convert labels to one-hot encoding
    y = to_categorical(y, num_classes=5)
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42
    )
    
    # Create and train model
    model = create_model((X.shape[1], X.shape[2]))
    
    # Add early stopping to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Add learning rate reduction on plateau
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.0001
    )
    
    # Training
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    # Save training configuration
    config = {
        'use_acc': use_acc,
        'features': data_loader.features,
        'gesture_map': data_loader.gesture_map,
        'sampling_rate': data_loader.sampling_rate,
        'sequence_length': data_loader.expected_samples
    }
    
    # Save model and config
    save_path = Path(base_path) / 'trained_model'
    save_path.mkdir(exist_ok=True)
    
    model.save(str(save_path / 'gesture_model.h5'))
    with open(save_path / 'config.json', 'w') as f:
        json.dump(config, f)
    
    print(f"\nModel and configuration saved to: {save_path}")
    
    return model, history, config

def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    base_path = r"C:\Users\Deker\Desktop\The Void\Python\SmartSenseML\roberto"  # Folder containing UP, DOWN, LEFT, RIGHT, REST folders
    
    try:
        # Train model with only gyro data
        model, history, config = train_model(
            base_path,
            use_acc=True,  # Only use gyro data
            rest_samples=10,  # Number of REST samples to use
            epochs=35,
            batch_size=32
        )
        
        # Plot training history
        plot_training_history(history)
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise