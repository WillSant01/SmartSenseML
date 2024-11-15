from customTskin import CustomTskin, Hand, OneFingerGesture
import time
import pandas as pd
from datetime import datetime

if __name__ == "__main__":
    timestamps = []
    acc_x_values = []
    acc_y_values = []
    acc_z_values = []
    gyro_x_values = []
    gyro_y_values = []
    gyro_z_values = []
    
    RECORDING_DURATION = 60
    SAMPLE_INTERVAL = 0.05
    last_sample_time = time.time()

    # 2 sec execution, 3 seconds pause
    # 40 lines for gesture, but in the autoencoder put 50 for contingecy
    # when recording starts, stay put for 2 seconds
    # mean of 11 gesture for csv, make 5 csv for gesture
    # 5 gestures; up, down, left, right, ok.
    with CustomTskin("C0:83:43:39:21:57", Hand.RIGHT) as tskin: #remember to change the 23
        print("Starting 60-second data collection...")
        start_time = time.time()
        
        while True:
            if not tskin.connected:
                print("Connecting..")
                time.sleep(0.1)
                continue
                
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            if elapsed_time >= RECORDING_DURATION:
                break
                    
            if current_time - last_sample_time >= SAMPLE_INTERVAL:
                gyro = tskin.gyro
                acc = tskin.acceleration
                
                if gyro and acc:
                    current_timestamp = datetime.fromtimestamp(current_time).strftime('%H:%M:%S.%f')[:-4]
                    
                    timestamps.append(current_timestamp)
                    gyro_x_values.append(round(gyro.x, 5))
                    gyro_y_values.append(round(gyro.y, 5))
                    gyro_z_values.append(round(gyro.z, 5))

                    acc_x_values.append(round(acc.x, 5))
                    acc_y_values.append(round(acc.y, 5))
                    acc_z_values.append(round(acc.z, 5))
                    
                    last_sample_time = current_time
                    print(f"Time: {current_timestamp}, GyroX: {gyro_x_values[-1]}, GyroY: {gyro_y_values[-1]}, GyroZ: {gyro_z_values[-1]}")
                
            time.sleep(tskin.TICK)
        
        df = pd.DataFrame({
            'Timestamp': timestamps,
            'AccX': acc_x_values,
            'AccY': acc_y_values,
            'AccZ': acc_z_values,
            'GyroX': gyro_x_values,
            'GyroY': gyro_y_values,
            'GyroZ': gyro_z_values
        })
        
        print(f"\nGyroscope data collected over {RECORDING_DURATION} seconds ({SAMPLE_INTERVAL}s intervals):")
        pd.set_option('display.float_format', lambda x: '%.5f' % x)
        print(df.to_string(index=False))
        
        #df.to_csv(r"roberto/RIGHT/right10.csv")