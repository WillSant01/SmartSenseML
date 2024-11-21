from customTskin import CustomTskin, Hand
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque
import json
from pathlib import Path
import time

class GestureDetector:
    def __init__(self, model_path, window_size=35):
        self.model = load_model(str(model_path / 'gesture_model.h5'))
       
        with open(model_path / 'config.json', 'r') as f:
            config = json.load(f)
            self.features = config['features']
            self.gesture_map = {int(v): k for k, v in config['gesture_map'].items()}
       
        self.window_size = window_size
        self.sensor_buffer = deque(maxlen=window_size)

    def add_reading(self, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z):
        sensor_data = []
        for feature in self.features:
            if feature == 'GyroX': sensor_data.append(gyro_x)
            elif feature == 'GyroY': sensor_data.append(gyro_y)
            elif feature == 'GyroZ': sensor_data.append(gyro_z)
            elif feature == 'AccX': sensor_data.append(acc_x)
            elif feature == 'AccY': sensor_data.append(acc_y)
            elif feature == 'AccZ': sensor_data.append(acc_z)
           
        self.sensor_buffer.append(sensor_data)
       
        if len(self.sensor_buffer) == self.window_size:
            window_data = np.array(self.sensor_buffer)
            prediction = self.model.predict(window_data.reshape(1, self.window_size, len(self.features)), verbose=0)
            gesture_idx = np.argmax(prediction[0])
            confidence = prediction[0][gesture_idx] * 100
           
            if confidence > 85 and self.gesture_map[gesture_idx] != '0':
                return self.gesture_map[gesture_idx], confidence
           
            self.sensor_buffer.popleft()
        return None, None

if __name__ == "__main__":
    model_path = Path(r"C:\Users\WilliamSanteramo\OneDrive - ITS Angelo Rizzoli\Documenti\UFS\15 IoT\SmartSenseML\roberto\trained_model")
    detector = GestureDetector(model_path)
   
    with CustomTskin("C0:83:3E:39:21:57", Hand.RIGHT) as tskin:
        last_sample_time = time.time()
       
        while True:
            if not tskin.connected:
                time.sleep(0.1)
                continue
           
            current_time = time.time()
            if current_time - last_sample_time >= 0.05:
                acc = tskin.acceleration
                gyro = tskin.gyro
               
                if acc and gyro:
                    gesture, confidence = detector.add_reading(
                        acc.x, acc.y, acc.z,
                        gyro.x, gyro.y, gyro.z
                    )
                   
                    if gesture:
                        print(f"{gesture}: {confidence:.1f}%")
               
                last_sample_time = current_time
           
            time.sleep(0.001)