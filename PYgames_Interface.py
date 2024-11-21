import pygame
import numpy as np
import time
from customTskin import CustomTskin, Hand
from tensorflow.keras.models import load_model
from collections import deque
import os
import json

class GestureGame:
    def __init__(self, model_dir):
        # Initialize PyGame
        pygame.init()
        
        # Window settings
        self.WINDOW_SIZE = 600
        self.CELL_SIZE = self.WINDOW_SIZE // 3
        self.screen = pygame.display.set_mode((self.WINDOW_SIZE, self.WINDOW_SIZE + 100))  # Extra space for gesture info
        pygame.display.set_caption("Gesture Control Grid")
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GRAY = (128, 128, 128)
        self.BLUE = (0, 0, 255)
        self.GREEN = (0, 255, 0)
        
        # Game state
        self.current_position = [1, 1]  # Start in center
        self.last_gesture = None
        self.last_confidence = 0
        self.can_move = True
        
        # Load model and config
        self.load_model(model_dir)
        
        # Sensor data buffer
        self.window_size = 50  # Match training window size
        self.sensor_buffer = deque(maxlen=self.window_size)
        
        # Initialize CustomTskin
        self.tskin = CustomTskin("C0:83:43:39:21:57", Hand.RIGHT)
        self.tskin.connect()
        
        # Timing control
        self.SAMPLE_INTERVAL = 0.05  # 50ms between samples (20Hz)
        self.last_sample_time = time.time()
        self.last_gesture_time = 0
        self.GESTURE_COOLDOWN = 1.0  # Wait 1 second between gestures
        
    def load_model(self, model_dir):
        """Load the trained model and configuration."""
        self.model = load_model(os.path.join(model_dir, 'gesture_model.h5'))
        with open(os.path.join(model_dir, 'config.json'), 'r') as f:
            self.config = json.load(f)
        self.gesture_map = {int(v): k for k, v in self.config['gesture_map'].items()}
        
    def process_gesture(self, gesture, confidence):
        """Process detected gesture and update game state."""
        if confidence < 0.6:  # 60% confidence threshold
            return
            
        message = f"Gesture: {gesture} ({confidence:.2f})"
        new_pos = self.current_position.copy()
        
        if gesture == "UP" and self.current_position[1] > 0:
            new_pos[1] -= 1
            self.current_position = new_pos
        elif gesture == "DOWN" and self.current_position[1] < 2:
            new_pos[1] += 1
            self.current_position = new_pos
        elif gesture == "LEFT" and self.current_position[0] > 0:
            new_pos[0] -= 1
            self.current_position = new_pos
        elif gesture == "RIGHT" and self.current_position[0] < 2:
            new_pos[0] += 1
            self.current_position = new_pos
            
        if new_pos == self.current_position:
            message = f"You've hit a Wall! {message}"
            
        self.last_gesture = message
        self.last_confidence = confidence
        
    def draw(self):
        """Draw the game state."""
        # Clear screen
        self.screen.fill(self.WHITE)
        
        # Draw grid
        for i in range(4):
            pygame.draw.line(self.screen, self.BLACK, 
                           (i * self.CELL_SIZE, 0),
                           (i * self.CELL_SIZE, self.WINDOW_SIZE))
            pygame.draw.line(self.screen, self.BLACK,
                           (0, i * self.CELL_SIZE),
                           (self.WINDOW_SIZE, i * self.CELL_SIZE))
        
        # Draw current position
        x = self.current_position[0] * self.CELL_SIZE
        y = self.current_position[1] * self.CELL_SIZE
        pygame.draw.rect(self.screen, self.BLUE,
                        (x + 2, y + 2, self.CELL_SIZE - 4, self.CELL_SIZE - 4))
        
        # Draw gesture information
        if self.last_gesture:
            font = pygame.font.Font(None, 36)
            text = font.render(self.last_gesture, True, self.BLACK)
            self.screen.blit(text, (10, self.WINDOW_SIZE + 30))
        
        pygame.display.flip()
        
    def process_sensor_data(self):
        """Process sensor data and detect gestures."""
        if len(self.sensor_buffer) == self.window_size:
            # Prepare data for model
            data = np.array(self.sensor_buffer)
            model_input = data.reshape(1, self.window_size, 3)  # 3 features (GyroX, GyroY, GyroZ)
            
            # Get prediction
            prediction = self.model.predict(model_input, verbose=0)
            gesture_idx = np.argmax(prediction[0])
            confidence = prediction[0][gesture_idx]
            
            # Process gesture if confidence threshold met
            gesture = self.gesture_map[gesture_idx]
            if gesture != "REST" and confidence >= 0.6:
                self.process_gesture(gesture, confidence)
                
            # Clear buffer for next gesture
            self.sensor_buffer.clear()
    
    def run(self):
        """Main game loop."""
        running = True
        while running:
            # Handle PyGame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    
            # Get sensor data
            current_time = time.time()
            if current_time - self.last_sample_time >= self.SAMPLE_INTERVAL:
                if not self.tskin.connected:
                    print("Reconnecting to sensor...")
                    self.tskin.connect()
                    continue
                    
                # Get gyroscope data
                gyro = self.tskin.gyroscope
                if gyro:
                    self.sensor_buffer.append([gyro.x, gyro.y, gyro.z])
                    self.last_sample_time = current_time
                    
                    # Process data if we have enough samples
                    self.process_sensor_data()
            
            # Update display
            self.draw()
            
            # Small sleep to prevent CPU overload
            time.sleep(0.001)
            
        # Cleanup
        self.tskin.disconnect()
        pygame.quit()

if __name__ == "__main__":
    model_dir = "trained_model"  # Update with your model directory
    game = GestureGame(model_dir)
    game.run()