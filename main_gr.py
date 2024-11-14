from customTskin import CustomTskin, Hand, OneFingerGesture
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque
import joblib
import json
import time
import os

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
        
        # Process when we have a full window and enough new samples
        if len(self.sensor_buffer) == self.window_size:
            # Check if we have 5 new samples (0.25s worth of data)
            new_samples = len(self.sensor_buffer) - self.last_processed_size
            if new_samples >= 5:  # Process every 0.25 seconds
                window_data = np.array(self.sensor_buffer)
                
                # Normalize data using the same scaler used during training
                normalized_data = self.scaler.transform(window_data)
                
                # Prepare data for model input (shape: [1, 50, 6])
                model_input = normalized_data.reshape(1, self.window_size, 6)
                
                # Check against each gesture model
                detections = {}
                for gesture_name, model in self.models.items():
                    # Get reconstruction
                    prediction = model.predict(model_input, verbose=0)
                    
                    # Calculate reconstruction error
                    error = np.mean(np.abs(model_input - prediction))
                    
                    # Compare against threshold
                    if error > self.thresholds[gesture_name]:
                        confidence = (error - self.thresholds[gesture_name]) / error
                        detections[gesture_name] = confidence
                
                # Remove 5 oldest samples (0.25s worth)
                for _ in range(5):
                    self.sensor_buffer.popleft()
                self.last_processed_size = len(self.sensor_buffer)
                        
                return detections
        
        return {}

if __name__ == "__main__":
    # Initialize gesture detector with trained models
    detector = GestureDetector("trained_models")  # Update with your model directory
    
    # Initialize CustomTskin
    with CustomTskin("C0:83:43:39:21:57", Hand.RIGHT) as tskin:
        print("Starting gesture detection...")
        print("Collecting initial window of sensor data (2.5 seconds)...")
        
        # Timing control
        SAMPLE_INTERVAL = 0.05  # 50ms between samples, matching training data
        last_sample_time = time.time()
        
        # For smoothing detections
        last_detection_time = 0
        DETECTION_COOLDOWN = 0.5  # Reduced cooldown since we have better coverage
        
        while True:
            if not tskin.connected:
                print("Connecting..")
                time.sleep(0.1)
                continue
            
            current_time = time.time()
            
            # Only sample at the specified interval (50ms)
            if current_time - last_sample_time >= SAMPLE_INTERVAL:
                # Get acceleration and gyroscope data
                acc = tskin.acceleration
                gyro = tskin.gyroscope
                
                if acc and gyro:
                    # Add reading to gesture detector
                    detections = detector.add_reading(
                        acc.x, acc.y, acc.z,
                        gyro.x, gyro.y, gyro.z
                    )
                    
                    # If gestures detected and cooldown passed, print them
                    if detections and (current_time - last_detection_time) >= DETECTION_COOLDOWN:
                        # Get gesture with highest confidence
                        gesture = max(detections.items(), key=lambda x: x[1])
                        print(f"Detected: {gesture[0].upper()} (confidence: {gesture[1]:.2f})")
                        last_detection_time = current_time
                
                last_sample_time = current_time
            
            # Small sleep to prevent CPU overload while maintaining timing
            time.sleep(0.001)