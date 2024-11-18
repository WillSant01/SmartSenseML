import numpy as np
from tensorflow.keras.models import load_model
from collections import deque, defaultdict
import joblib
import os
import pandas as pd

class GestureDetector:
    def __init__(self, model_dir, window_size=50):
        self.window_size = window_size
        self.sensor_buffer = deque(maxlen=window_size)
        self.gesture_map = {
            0: 'rest',
            1: 'up',
            2: 'down',
            3: 'left',
            4: 'right'
        }
        self.cooldown = 25  # Cooldown period (about half the min gesture length)
        self.cooldown_counter = 0
        self.last_gesture = None
        
        # Load model and scaler
        self.model = load_model(os.path.join(model_dir, "gesture_model.h5"))
        self.scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
            
    def add_reading(self, gyro_x, gyro_y, gyro_z):
        """Add a new sensor reading to the buffer and check for gestures"""
        # Format sensor data
        sensor_data = np.array([gyro_x, gyro_y, gyro_z])
        self.sensor_buffer.append(sensor_data)
        
        # Process when we have a full window
        if len(self.sensor_buffer) == self.window_size:
            window_data = np.array(self.sensor_buffer)
            
            # Normalize data using the same scaler used during training
            normalized_data = self.scaler.transform(window_data)
            model_input = normalized_data.reshape(1, self.window_size, 3)
            
            # Get prediction
            prediction = self.model.predict(model_input, verbose=0)
            pred_class = np.argmax(prediction[0])
            confidence = prediction[0][pred_class]
            
            # Handle cooldown
            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1
                return None
            
            # Return gesture and confidence if not rest, or confidence high enough
            if pred_class != 0 and confidence > 0.90:  # Slightly lower threshold but no rest
                self.cooldown_counter = self.cooldown
                self.last_gesture = self.gesture_map[pred_class]
                return {
                    'gesture': self.gesture_map[pred_class],
                    'confidence': float(confidence),
                    'predictions': {self.gesture_map[i]: float(p) for i, p in enumerate(prediction[0])}
                }
            
        return None

def cluster_detections(detections, min_gap=25):
    """Cluster nearby detections of the same gesture"""
    if not detections:
        return []
    
    clustered = []
    current_cluster = {
        'gesture': detections[0]['gesture'],
        'start_frame': 0,
        'end_frame': 0,
        'confidences': [detections[0]['confidence']],
        'raw_detections': 1
    }
    
    last_frame = 0
    
    for i, det in enumerate(detections):
        if (i > last_frame + min_gap) or (det['gesture'] != current_cluster['gesture']):
            # Save current cluster
            current_cluster['avg_confidence'] = np.mean(current_cluster['confidences'])
            clustered.append(current_cluster)
            
            # Start new cluster
            current_cluster = {
                'gesture': det['gesture'],
                'start_frame': i,
                'end_frame': i,
                'confidences': [det['confidence']],
                'raw_detections': 1
            }
        else:
            # Add to current cluster
            current_cluster['end_frame'] = i
            current_cluster['confidences'].append(det['confidence'])
            current_cluster['raw_detections'] += 1
            
        last_frame = i
    
    # Add final cluster
    current_cluster['avg_confidence'] = np.mean(current_cluster['confidences'])
    clustered.append(current_cluster)
    
    return clustered

def analyze_gesture_flow(all_detections, total_frames):
    """Analyze the flow of gestures in the data"""
    # Create frame-by-frame gesture mapping
    frame_gestures = ['rest'] * total_frames
    
    for det in all_detections:
        start = det['start_frame']
        end = det['end_frame']
        gesture = det['gesture']
        for i in range(start, end + 1):
            if i < total_frames:
                frame_gestures[i] = gesture
    
    # Find transitions
    transitions = []
    current_gesture = frame_gestures[0]
    start_frame = 0
    
    for i, gesture in enumerate(frame_gestures):
        if gesture != current_gesture:
            transitions.append({
                'gesture': current_gesture,
                'start': start_frame,
                'duration': i - start_frame
            })
            current_gesture = gesture
            start_frame = i
    
    # Add final transition
    transitions.append({
        'gesture': current_gesture,
        'start': start_frame,
        'duration': total_frames - start_frame
    })
    
    return transitions

def test_with_csv(csv_path, model_dir, window_size=50):
    """Test the gesture detector with CSV data"""
    # Load CSV data
    data = pd.read_csv(csv_path)
    print(f"CSV length: {len(data)} samples ({len(data)/20:.1f} seconds)")
    
    # Initialize detector
    detector = GestureDetector(model_dir, window_size)
    
    # Storage for analysis
    all_detections = []
    raw_detection_count = 0
    
    # Process each row
    for idx, row in data.iterrows():
        detection = detector.add_reading(
            row['GyroX'], row['GyroY'], row['GyroZ']
        )
        
        if detection:
            detection['frame'] = idx
            all_detections.append(detection)
            raw_detection_count += 1
    
    # Cluster detections
    clustered = cluster_detections(all_detections)
    
    # Analyze gesture flow
    gesture_flow = analyze_gesture_flow(clustered, len(data))
    
    # Print Analysis
    print("\nGesture Flow Analysis:")
    print("-----------------")
    print(f"Total length: {len(data)/20:.1f} seconds ({len(data)} samples)")
    print(f"Raw detections: {raw_detection_count}")
    print(f"Clustered gestures: {len(clustered)}")
    
    print("\nDetected Gesture Sequence:")
    print("--------------------------")
    for i, transition in enumerate(gesture_flow):
        if transition['gesture'] != 'rest' or transition['duration'] > window_size:
            start_time = transition['start'] / 20  # Convert to seconds
            duration = transition['duration'] / 20
            print(f"{i+1}. {transition['gesture']:>5} : {start_time:>5.1f}s - {start_time+duration:>5.1f}s (duration: {duration:>4.1f}s)")
    
    print("\nClustered Gestures Details:")
    print("-------------------------")
    for i, cluster in enumerate(clustered):
        start_time = cluster['start_frame'] / 20
        end_time = cluster['end_frame'] / 20
        duration = end_time - start_time
        print(f"{i+1}. {cluster['gesture']:>5} : {start_time:>5.1f}s - {end_time:>5.1f}s "
              f"(duration: {duration:>4.1f}s, confidence: {cluster['avg_confidence']:.3f})")
    
    return clustered, gesture_flow

if __name__ == "__main__":
    # Update these paths for your setup
    csv_path = r"C:\Users\Deker\Desktop\The Void\Python\SmartSenseML\roberto\DOWN\l_down7.csv"
    model_dir = r"C:\Users\Deker\Desktop\The Void\Python\SmartSenseML\roberto\trained_model_20241118_224744"
    
    print(f"Testing gesture detection on {csv_path}")
    clustered_detections, flow = test_with_csv(csv_path, model_dir)