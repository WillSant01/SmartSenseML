import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Layer, Dense
import json

class AttentionWithMasking(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score_dense = Dense(1, activation='tanh')

    def call(self, inputs, mask=None):
        score = self.score_dense(inputs)
        score = tf.squeeze(score, axis=-1)
        
        if mask is not None:
            score += (1.0 - tf.cast(mask, dtype=score.dtype)) * -1e9
            
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = tf.reduce_sum(inputs * tf.expand_dims(attention_weights, -1), axis=1)
        return context_vector, attention_weights

    def compute_mask(self, inputs, mask=None):
        return None

class GestureProcessor:
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self._load_model()
        self._load_config()
        
    def _load_model(self):
        try:
            self.model = load_model(
                str(self.model_path / 'gesture_model.h5'),
                custom_objects={'AttentionWithMasking': AttentionWithMasking}
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def _load_config(self):
        try:
            with open(self.model_path / 'config.json', 'r') as f:
                self.config = json.load(f)
            self.labels = {v: k for k, v in self.config['gesture_map'].items()}
            self.features = self.config['features']
            self.sequence_length = self.config['sequence_length']
            self.sampling_rate = self.config['sampling_rate']
        except Exception as e:
            raise RuntimeError(f"Failed to load config: {e}")

    def process_file(self, file_path, threshold=0.1):
        """Process a single gesture file."""
        try:
            # Data loading and validation
            data = pd.read_csv(file_path)
            missing = set(self.features) - set(data.columns)
            if missing:
                raise ValueError(f"Missing features: {missing}")

            # Extract features and detect motion
            feature_data = data[self.features].values
            motion = self._detect_motion(feature_data, threshold)
            
            # Segment gestures
            segments = self._segment_gestures(feature_data, motion)
            if not segments:
                return pd.DataFrame()
                
            # Make predictions
            results = self._predict_segments(segments)
            return results
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return pd.DataFrame()

    def _detect_motion(self, data, threshold):
        """Detect motion in sensor data."""
        gyro_motion = np.sqrt(
            data[:, self.features.index('GyroX')]**2 +
            data[:, self.features.index('GyroY')]**2 +
            data[:, self.features.index('GyroZ')]**2
        )
        return gyro_motion > threshold

    def _segment_gestures(self, data, motion, min_length=35):
        """Segment data into gesture windows."""
        segments = []
        timestamps = []
        
        # Find motion transitions
        transitions = np.diff(motion.astype(int))
        starts = np.where(transitions == 1)[0]
        ends = np.where(transitions == -1)[0]
        
        # Handle boundary cases
        if len(starts) == 0 or len(ends) == 0:
            return None
            
        if starts[0] > ends[0]:
            ends = ends[1:]
        if len(starts) > len(ends):
            starts = starts[:-1]
            
        # Process segments
        for start, end in zip(starts, ends):
            if end - start >= min_length:
                segment = data[start:min(end, start + self.sequence_length)]
                if len(segment) < self.sequence_length:
                    padding = np.zeros((self.sequence_length - len(segment), len(self.features)))
                    segment = np.vstack([segment, padding])
                segments.append(segment)
                timestamps.append(start / self.sampling_rate)
                
        return {'segments': np.array(segments), 'timestamps': np.array(timestamps)} if segments else None

    def _predict_segments(self, segment_data):
        """Make predictions on segmented data."""
        predictions = self.model.predict(segment_data['segments'], verbose=0)
        
        results = pd.DataFrame({
            'Time': segment_data['timestamps'],
            'Gesture': [self.labels[np.argmax(p)] for p in predictions],
            'Confidence': [np.max(p) for p in predictions]
        })
        
        return results[results['Confidence'] > 0.5]  # Filter low confidence predictions

    def visualize_results(self, results):
        """Visualize prediction results."""
        if results.empty:
            print("No predictions to visualize")
            return
            
        plt.figure(figsize=(12, 6))
        
        # Plot gestures
        plt.subplot(2, 1, 1)
        unique_gestures = list(self.labels.values())
        gesture_nums = [unique_gestures.index(g) for g in results['Gesture']]
        
        plt.scatter(results['Time'], gesture_nums, c=results['Confidence'], 
                   cmap='viridis', s=100)
        plt.colorbar(label='Confidence')
        plt.yticks(range(len(unique_gestures)), unique_gestures)
        plt.xlabel('Time (s)')
        plt.ylabel('Gesture')
        plt.title('Detected Gestures')
        
        # Plot confidence
        plt.subplot(2, 1, 2)
        plt.plot(results['Time'], results['Confidence'], 'b-')
        plt.xlabel('Time (s)')
        plt.ylabel('Confidence')
        plt.title('Prediction Confidence')
        
        plt.tight_layout()
        plt.show()

def main():
    model_path = r"C:\Users\Deker\Desktop\The Void\Python\SmartSenseML\roberto\trained_model"
    processor = GestureProcessor(model_path)
    
    test_file = r"C:\Users\Deker\Desktop\The Void\Python\SmartSenseML\roberto\LEFT\l_left6.csv"
    results = processor.process_file(test_file)
    
    if not results.empty:
        print("\nDetected Gestures:")
        print(results.to_string(index=False))
        processor.visualize_results(results)
    else:
        print("No gestures detected")

if __name__ == "__main__":
    main()