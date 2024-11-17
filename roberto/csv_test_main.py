import numpy as np
from tensorflow.keras.models import load_model
from collections import deque
import joblib
import json
import os
import pandas as pd

class GestureDetector:
    def __init__(self, model_dir, window_size=50):
        """
        Initialize gesture detector with trained models
        
        Args:
            model_dir: Directory containing trained models and parameters
            window_size: Size of the sliding window for gesture detection (50 samples = 2.5s)
        """
        self.window_size = window_size
        self.sensor_buffer = deque(maxlen=window_size)
        self.last_processed_size = 0
        
        # Load models and parameters
        self.scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
        with open(os.path.join(model_dir, "thresholds.json"), 'r') as f:
            self.thresholds = json.load(f)
            
        self.models = {}
        for gesture in self.thresholds.keys():
            model_path = os.path.join(model_dir, f"model_{gesture}.h5")
            self.models[gesture] = load_model(model_path)
            
    def add_reading(self, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z):
        """Add a new sensor reading to the buffer and check for gestures"""
        # Format sensor data
        sensor_data = np.array([acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z])
        self.sensor_buffer.append(sensor_data)
        
        # Process when we have a full window
        if len(self.sensor_buffer) == self.window_size:
            window_data = np.array(self.sensor_buffer)
            
            # Normalize data using the same scaler used during training
            normalized_data = self.scaler.transform(window_data)
            
            # Prepare data for model input (shape: [1, 50, 6])
            model_input = normalized_data.reshape(1, self.window_size, 6)
            
            # Check against each gesture model
            detections = {}
            reconstructions = {}
            for gesture_name, model in self.models.items():
                # Get reconstruction
                prediction = model.predict(model_input, verbose=0)
                
                # Calculate reconstruction error
                error = np.mean(np.abs(model_input - prediction))
                
                # Store reconstruction for analysis
                reconstructions[gesture_name] = prediction
                
                # Compare against threshold
                if error > self.thresholds[gesture_name]:
                    confidence = (error - self.thresholds[gesture_name]) / error
                    detections[gesture_name] = confidence
            
            # Return both detections and reconstructions for analysis
            return detections, reconstructions, model_input
        
        return {}, None, None

def test_with_csv(csv_path, model_dir, known_gesture=None):
    """
    Test the gesture detector with CSV data
    
    Args:
        csv_path: Path to CSV file containing sensor data
        model_dir: Directory containing trained models
        known_gesture: If provided, the actual gesture in the CSV for validation
    """
    # Load CSV data
    data = pd.read_csv(csv_path)
    
    # Map column names to match training data
    column_mapping = {
        'AccX': 'acc_x',
        'AccY': 'acc_y',
        'AccZ': 'acc_z',
        'GyroX': 'gyro_x',
        'GyroY': 'gyro_y',
        'GyroZ': 'gyro_z'
    }
    
    # Rename columns if they don't match expected names
    if 'AccX' in data.columns:
        data = data.rename(columns=column_mapping)
    
    # Initialize detector
    detector = GestureDetector(model_dir)
    
    # Storage for analysis
    all_detections = []
    all_reconstructions = {}
    all_inputs = []
    
    # Process each row
    for idx, row in data.iterrows():
        detections, reconstructions, model_input = detector.add_reading(
            row['acc_x'], row['acc_y'], row['acc_z'],
            row['gyro_x'], row['gyro_y'], row['gyro_z']
        )
        
        if detections:
            all_detections.append(detections)
            if reconstructions:
                # Initialize storage for first reconstruction
                if not all_reconstructions:
                    all_reconstructions = {gesture: [] for gesture in reconstructions.keys()}
                # Store reconstructions
                for gesture, recon in reconstructions.items():
                    all_reconstructions[gesture].append(recon)
                all_inputs.append(model_input)
    
    # Analysis
    print("\nDetection Analysis:")
    print("-----------------")
    
    # Count total windows analyzed
    total_windows = len(all_inputs) if all_inputs else 0
    print(f"Total windows analyzed: {total_windows}")
    
    if total_windows == 0:
        print("No complete windows were processed. Check if data format matches expectations.")
        return [], {}, []
    
    # Analyze detections per gesture
    detection_counts = {}
    for detections in all_detections:
        for gesture, confidence in detections.items():
            if gesture not in detection_counts:
                detection_counts[gesture] = []
            detection_counts[gesture].append(confidence)
    
    for gesture, confidences in detection_counts.items():
        count = len(confidences)
        avg_confidence = np.mean(confidences)
        print(f"\nGesture: {gesture}")
        print(f"Detection count: {count} ({count/total_windows*100:.1f}% of windows)")
        print(f"Average confidence: {avg_confidence:.3f}")
        
        if known_gesture:
            if gesture == known_gesture:
                print(f"True Positive Rate: {count/total_windows*100:.1f}%")
            else:
                print(f"False Positive Rate: {count/total_windows*100:.1f}%")
    
    return all_detections, all_reconstructions, all_inputs

if __name__ == "__main__":
    # Update these paths for your setup
    csv_path = r"C:\Users\Deker\Downloads\test_iot.csv"  # Your CSV file with sensor data
    model_dir = r"C:\Users\Deker\Desktop\Autoencoder\roberto\trained_models_20241117_172401"  # Directory with trained models
    
    # Optional: specify the known gesture in the CSV for validation
    known_gesture = "up"  # Or None if unknown
    
    print(f"Testing gesture detection on {csv_path}")
    print(f"Expected gesture: {known_gesture if known_gesture else 'Unknown'}")
    
    detections, reconstructions, inputs = test_with_csv(csv_path, model_dir, known_gesture)