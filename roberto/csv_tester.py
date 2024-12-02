import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import json
from typing import Dict, List, Optional, Tuple, Union
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

class GestureProcessor:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model = None
        self.config = None
        self.gesture_map = None
        self.features = None
        self._load_model_and_config()

    def _load_model_and_config(self) -> None:
        """Load the trained model and configuration."""
        try:
            self.model = load_model(str(self.model_path / 'gesture_model.h5'))
            
            with open(self.model_path / 'config.json', 'r') as f:
                self.config = json.load(f)
            
            self.gesture_map = self.config['gesture_map']
            self.features = self.config['features']
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model or config: {e}")

    def process_file(self, file_path: str, threshold: float = 0.1, 
                    min_gesture_length: int = 35) -> pd.DataFrame:
        """Process a single file containing gesture data."""
        try:
            data = pd.read_csv(file_path)
            if not set(self.features).issubset(set(data.columns)):
                raise ValueError(f"Missing required features: {set(self.features) - set(data.columns)}")

            feature_data = data[self.features].values
            motion_segments = self._detect_and_segment(feature_data, threshold, min_gesture_length)
            
            if not motion_segments:
                return pd.DataFrame()

            predictions = self._make_predictions(motion_segments['segments'])
            
            results = pd.DataFrame({
                'timestamp': motion_segments['timestamps'],
                'gesture': [self._idx_to_gesture(np.argmax(p)) for p in predictions],
                'confidence': [float(np.max(p)) for p in predictions]
            })

            if 'label' in data.columns:
                results['actual'] = self._get_actual_labels(data, motion_segments['timestamps'])

            return results

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return pd.DataFrame()

    def _detect_and_segment(self, data: np.ndarray, threshold: float, 
                          min_length: int) -> Optional[Dict[str, np.ndarray]]:
        """Detect motion and segment into potential gestures."""
        # Calculate motion intensity using gyroscope data
        gyro_indices = [i for i, f in enumerate(self.features) if f.startswith('Gyro')]
        motion = np.sqrt(np.sum(data[:, gyro_indices] ** 2, axis=1)) > threshold

        # Find motion segments
        transitions = np.diff(motion.astype(int))
        starts = np.where(transitions == 1)[0]
        ends = np.where(transitions == -1)[0]

        if len(starts) == 0 or len(ends) == 0:
            return None

        # Adjust boundary cases
        if starts[0] > ends[0]:
            ends = ends[1:]
        if len(starts) > len(ends):
            starts = starts[:-1]

        segments = []
        timestamps = []

        for start, end in zip(starts, ends):
            if end - start >= min_length:
                segment = data[start:end]
                if len(segment) >= min_length:
                    segments.append(segment)
                    timestamps.append(start)

        if not segments:
            return None

        # Pad sequences to same length
        max_len = max(len(s) for s in segments)
        padded_segments = np.zeros((len(segments), max_len, data.shape[1]))
        for i, seg in enumerate(segments):
            padded_segments[i, :len(seg)] = seg

        return {
            'segments': padded_segments,
            'timestamps': np.array(timestamps)
        }

    def _make_predictions(self, segments: np.ndarray) -> np.ndarray:
        """Make predictions on segmented data."""
        return self.model.predict(segments, verbose=0)

    def _idx_to_gesture(self, idx: int) -> str:
        """Convert prediction index to gesture name."""
        reverse_map = {int(v): k for k, v in self.gesture_map.items()}
        return reverse_map.get(idx, 'unknown')

    def _get_actual_labels(self, data: pd.DataFrame, timestamps: np.ndarray) -> List[str]:
        """Get actual labels for timestamps."""
        return [str(data['label'].iloc[t]) if t < len(data) else 'unknown' 
                for t in timestamps]

    def visualize_results(self, results: pd.DataFrame) -> None:
        """Visualize prediction results."""
        if results.empty:
            print("No predictions to visualize")
            return

        plt.figure(figsize=(12, 8))

        # Plot gestures
        unique_gestures = sorted(set(self.gesture_map.keys()))
        gesture_nums = [unique_gestures.index(g) for g in results['gesture']]

        plt.subplot(2, 1, 1)
        scatter = plt.scatter(results['timestamp'], gesture_nums, 
                            c=results['confidence'], cmap='viridis', 
                            s=100, label='Predicted')

        if 'actual' in results.columns:
            actual_nums = [unique_gestures.index(g) if g in unique_gestures else -1 
                         for g in results['actual']]
            plt.scatter(results['timestamp'], actual_nums, marker='x', 
                       color='red', s=100, label='Actual')
            plt.legend()

        plt.colorbar(scatter, label='Confidence')
        plt.yticks(range(len(unique_gestures)), unique_gestures)
        plt.xlabel('Time')
        plt.ylabel('Gesture')
        plt.title('Detected Gestures')

        # Plot confidence
        plt.subplot(2, 1, 2)
        plt.plot(results['timestamp'], results['confidence'], 'b-')
        plt.axhline(y=0.5, color='r', linestyle='--', label='Confidence Threshold')
        plt.xlabel('Time')
        plt.ylabel('Confidence')
        plt.title('Prediction Confidence')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def print_metrics(self, results: pd.DataFrame) -> None:
        """Print performance metrics."""
        if 'actual' not in results.columns:
            return

        print("\nPerformance Metrics:")
        correct = (results['actual'] == results['gesture']).sum()
        total = len(results)
        print(f"Accuracy: {(correct/total)*100:.2f}% ({correct}/{total})")

        print("\nConfusion Matrix:")
        unique_labels = sorted(set(self.gesture_map.keys()))
        confusion = pd.DataFrame(0, index=unique_labels, columns=unique_labels)
        
        for actual, pred in zip(results['actual'], results['gesture']):
            if actual in unique_labels:
                confusion.loc[actual, pred] += 1
        
        print(confusion)

def main():
    """Main function to demonstrate usage."""
    model_path = r"C:\Users\WilliamSanteramo\OneDrive - ITS Angelo Rizzoli\Documenti\UFS\15 IoT\SmartSenseML\roberto\trained_model"  # Update with actual path
    test_file = r"C:\Users\WilliamSanteramo\OneDrive - ITS Angelo Rizzoli\Documenti\UFS\15 IoT\SmartSenseML\TEST_csv\test_df_ufficiale3.csv"   # Update with actual path

    try:
        processor = GestureProcessor(model_path)
        results = processor.process_file(test_file, threshold=0.1)
        
        if not results.empty:
            print("\nDetected Gestures:")
            print(results)
            processor.print_metrics(results)
            processor.visualize_results(results)
        else:
            print("No gestures detected")
            
    except Exception as e:
        print(f"Error in processing: {e}")

if __name__ == "__main__":
    main()