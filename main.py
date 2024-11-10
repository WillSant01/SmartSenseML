from customTskin import CustomTskin, Hand, OneFingerGesture
import time
import math

def calculate_acceleration_module(x, y, z):
    return math.sqrt(x**2 + y**2 + z**2)

if __name__ == "__main__":
    with CustomTskin("C0:83:43:39:21:57", Hand.RIGHT) as tskin:
        print("Starting acceleration monitoring...")
        print("Module indicators:")
        print("() : module ≤ 0.5")
        print("[] : 0.5 < module ≤ 1")
        print("{} : module > 1")
        
        while True:
            if not tskin.connected:
                print("Connecting..")
                time.sleep(0.1)
                continue
            
            acc = tskin.acceleration #finestra 100 ms (1° tentativo)
            if acc:
                acc_module = calculate_acceleration_module(acc.x, acc.y, acc.z)
                acc_module = round(acc_module, 5)
                
                if acc_module <= 0.7:
                    brackets = "()"
                elif acc_module <= 2.0:
                    brackets = "OOO"
                else:
                    breackets = "............."
                
                
                print(brackets)
            
            time.sleep(tskin.TICK)