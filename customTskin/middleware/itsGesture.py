from multiprocessing import Process, Event
from multiprocessing.connection import _ConnectionBase
import wave

class ITSGesture(Process):
    sensor_rx = _ConnectionBase

    def __init__(self, sensor_rx, audio_rx):
        Process.__init__(self)
        self.sensor_rx = sensor_rx
        self.audio_rx = audio_rx
        self.can_run = Event()

    def run(self):
        while self.can_run.is_set():
            if self.sensor_rx.poll():
                accX, accY, accZ, gyroX, gyroY, gyroZ = self.sensor_rx.recv()
                
                formatted_acc = (
                    round(accX, 5),
                    round(accY, 5),
                    round(accZ, 5)
                )

                formatted_gyro = (
                    round(gyroX, 5),
                    round(gyroY, 5),
                    round(gyroZ, 5)
                )

            if self.audio_rx.poll():
                with wave.open("test.wav", "wb") as audio_file:
                    audio_file.setnchannels(1)
                    audio_file.setframerate(16000)
                    audio_file.setsampwidth(2)

                    while self.audio_rx.poll(1):
                        audio_bytes = self.audio_rx.recv_bytes()
                        audio_file.writeframes(audio_bytes)