import random
import time

def get_simulated_distance():
    # Simulate distance between 30cm and 300cm
    return random.randint(30, 300)

if __name__ == "__main__":
    while True:
        print(f"Simulated Distance: {get_simulated_distance()} cm")
        time.sleep(1)
