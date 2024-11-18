import numpy as np
from tensorflow.keras.models import load_model
from collections import deque
import joblib
import os
import pandas as pd

class GestureDetector:
    def __init__(self, model_dir, window_size=50):
        """
        Initialize gesture detector with trained model
        
        Args:
            model_dir: Directory containing trained model and scaler
            window_size: Size of the sliding window for gesture detection (50 samples = 2.5s)
        """
        self.window_size = window_size
        self.sensor_buffer = deque(maxlen=window_size)
        self.gesture_map = {
            0: 'rest',
            1: 'up',
            2: 'down',
            3: 'left',
            4: 'right'
        }
        
        # Load model and scaler
        self.model = load_model(os.path.join(model_dir, "gesture_model.h5"))
        self.scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
            
    def add_reading(self, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z):
        """Add a new sensor reading to the buffer and check for gestures"""
        # Format sensor data (acc and gyro data)
        sensor_data = np.array([acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z])
        self.sensor_buffer.append(sensor_data)
        
        # Process when we have a full window
        if len(self.sensor_buffer) == self.window_size:
            window_data = np.array(self.sensor_buffer)
            
            # Normalize data using the same scaler used during training
            normalized_data = self.scaler.transform(window_data)
            
            # Prepare data for model input (shape: [1, 50, 6])
            model_input = normalized_data.reshape(1, self.window_size, 6)
            
            # Get prediction
            prediction = self.model.predict(model_input, verbose=0)
            pred_class = np.argmax(prediction[0])
            confidence = prediction[0][pred_class]
            
            # Return gesture and confidence if not rest, or confidence high enough
            if pred_class != 0 or confidence > 0.95:  # High threshold for rest to reduce false positives
                return {
                    'gesture': self.gesture_map[pred_class],
                    'confidence': float(confidence),
                    'predictions': {self.gesture_map[i]: float(p) for i, p in enumerate(prediction[0])}
                }
            
        return None

def test_with_csv(csv_path, model_dir, window_size=50):
    """
    Test the gesture detector with CSV data
    
    Args:
        csv_path: Path to CSV file containing sensor data
        model_dir: Directory containing trained model
    """
    # Load CSV data
    data = pd.read_csv(csv_path)
    
    # Initialize detector
    detector = GestureDetector(model_dir, window_size)
    
    # Storage for analysis
    all_detections = []
    detection_counts = {gesture: 0 for gesture in detector.gesture_map.values()}
    
    # Process each row
    for idx, row in data.iterrows():
        # Use both acc and gyro data
        detection = detector.add_reading(
            row['AccX'], row['AccY'], row['AccZ'],
            row['GyroX'], row['GyroY'], row['GyroZ']
        )
        
        if detection:
            all_detections.append(detection)
            detection_counts[detection['gesture']] += 1
            
            # Print real-time detection with confidence
            print(f"\nFrame {idx}")
            print(f"Detected: {detection['gesture']} (confidence: {detection['confidence']:.3f})")
            print("All probabilities:")
            for gesture, prob in detection['predictions'].items():
                print(f"  {gesture}: {prob:.3f}")
    
    # Analysis
    total_windows = len(all_detections)
    print("\nDetection Analysis:")
    print("-----------------")
    print(f"Total detection windows: {total_windows}")
    
    if total_windows > 0:
        print("\nGesture Distribution:")
        for gesture, count in detection_counts.items():
            if count > 0:
                print(f"{gesture}: {count} detections ({count/total_windows*100:.1f}% of detections)")
    else:
        print("No gestures detected in the data")
    
    return all_detections

if __name__ == "__main__":
    # Update these paths for your setup
    csv_path = r"C:\Users\Deker\Desktop\The Void\Python\SmartSenseML\roberto\DOWN\l_down7.csv"  # Your CSV file with sensor data
    model_dir = r"C:\Users\Deker\Desktop\The Void\Python\SmartSenseML\roberto\trained_model_20241118_222717"  # Directory with trained model
    
    print(f"Testing gesture detection on {csv_path}")
    detections = test_with_csv(csv_path, model_dir)