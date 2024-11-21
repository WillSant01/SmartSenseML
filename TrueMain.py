from customTskin import CustomTskin, Hand
import numpy as np
from tensorflow.keras.models import load_model
import json
from pathlib import Path
import time
import keyboard
from typing import Optional, Tuple, List

class GestureDetector:
    def __init__(self, model_path: Path, window_size: int = 35):
        self.model = load_model(str(model_path / 'gesture_model.h5'))
       
        with open(model_path / 'config.json', 'r') as f:
            config = json.load(f)
            self.features = config['features']
            self.gesture_map = {int(v): k for k, v in config['gesture_map'].items()}
       
        self.window_size = window_size
        self.readings: List[List[float]] = []
        self.recording = False
        self.threshold = 0.1
       
    def add_reading(self, acc_x: float, acc_y: float, acc_z: float,
                   gyro_x: float, gyro_y: float, gyro_z: float) -> Optional[Tuple[str, float]]:
        sensor_data = []
        for feature in self.features:
            if feature == 'GyroX': sensor_data.append(gyro_x)
            elif feature == 'GyroY': sensor_data.append(gyro_y)
            elif feature == 'GyroZ': sensor_data.append(gyro_z)
            elif feature == 'AccX': sensor_data.append(acc_x)
            elif feature == 'AccY': sensor_data.append(acc_y)
            elif feature == 'AccZ': sensor_data.append(acc_z)
       
        if self.recording:
            self.readings.append(sensor_data)
       
        return None, None
   
    def analyze_gesture(self) -> Optional[Tuple[str, float]]:
        print(f"Total readings collected: {len(self.readings)}")
       
        if len(self.readings) < self.window_size:
            print("Not enough readings")
            return None, None
           
        data = np.array(self.readings)
        gyro_indices = [i for i, f in enumerate(self.features) if f.startswith('Gyro')]
        motion = np.sqrt(np.sum(data[:, gyro_indices] ** 2, axis=1))
       
        print(f"Motion intensity range: {motion.min():.3f} to {motion.max():.3f}")
       
        active_motion = motion > self.threshold
        transitions = np.diff(active_motion.astype(int))
        starts = np.where(transitions == 1)[0]
        ends = np.where(transitions == -1)[0]
       
        print(f"Motion segments found: {len(starts)} starts, {len(ends)} ends")
       
        if len(starts) == 0 or len(ends) == 0:
            print("No motion segments detected")
            return None, None
           
        if starts[0] > ends[0]:
            ends = ends[1:]
        if len(starts) > len(ends):
            starts = starts[:-1]
           
        segment_lengths = ends - starts
        print(f"Segment lengths: {segment_lengths}")
       
        if not any(length >= self.window_size for length in segment_lengths):
            print("No segments long enough")
            return None, None
           
        longest_idx = np.argmax(segment_lengths)
        start, end = starts[longest_idx], ends[longest_idx]
       
        print(f"Selected segment: {start} to {end} (length: {end-start})")
       
        segment = data[start:end]
        if len(segment) > self.window_size:
            segment = segment[:self.window_size]
       
        prediction = self.model.predict(segment.reshape(1, self.window_size, len(self.features)), verbose=0)
        gesture_idx = np.argmax(prediction[0])
        confidence = prediction[0][gesture_idx] * 100
       
        print(f"Prediction confidence: {confidence:.1f}%")
       
        if confidence > 85 and self.gesture_map[gesture_idx] != '0':
            return self.gesture_map[gesture_idx], confidence
           
        return None, None
   
    def start_recording(self):
        self.recording = True
        self.readings = []
        print("Started recording gesture...")
       
    def stop_recording(self) -> Optional[Tuple[str, float]]:
        self.recording = False
        print("Stopped recording. Analyzing gesture...")
        return self.analyze_gesture()

if __name__ == "__main__":
    model_path = Path(r"C:\Users\WilliamSanteramo\OneDrive - ITS Angelo Rizzoli\Documenti\UFS\15 IoT\SmartSenseML\roberto\trained_model")
    detector = GestureDetector(model_path)
   
    with CustomTskin("C0:83:3E:39:21:57", Hand.RIGHT) as tskin:
        last_sample_time = time.time()
        print("Press SPACE to start/stop recording a gesture")
       
        recording_toggle = False
       
        while True:
            if not tskin.connected:
                time.sleep(0.1)
                continue
           
            if keyboard.is_pressed('space'):
                if not recording_toggle:
                    detector.start_recording()
                    recording_toggle = True
                else:
                    gesture, confidence = detector.stop_recording()
                    if gesture:
                        print(f"Detected gesture: {gesture} ({confidence:.1f}%)")
                    else:
                        print("No gesture detected")
                    recording_toggle = False
                time.sleep(0.2)  # Debounce
           
            current_time = time.time()
            if current_time - last_sample_time >= 0.05:  # 20Hz sampling
                acc = tskin.acceleration
                gyro = tskin.gyro
               
                if acc and gyro:
                    detector.add_reading(
                        acc.x, acc.y, acc.z,
                        gyro.x, gyro.y, gyro.z
                    )
               
                last_sample_time = current_time
           
            time.sleep(0.001)