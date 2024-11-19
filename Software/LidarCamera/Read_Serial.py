import serial
import time

# Update the port to match your setup
serial_port = '/dev/ttyACM0'  # Change this if necessary
baud_rate = 115200            # Match the baud rate in the Arduino code

try:
    ser = serial.Serial(serial_port, baud_rate)
    print(f"Connected to {serial_port} at {baud_rate} baud.")

    while True:
        if ser.in_waiting > 0:  # Check if data is available
            data = ser.readline().decode('utf-8', errors='ignore').strip()
            
            # Check if the line contains angle data (a float value)
            try:
                # Attempt to convert the data to a float (for angle)
                angle = float(data)
                print(f"Received angle: {angle} degrees")
            except ValueError:
                # If conversion fails, check for distance data (contains 'cm' in the string)
                if "cm" in data:
                    print(f"Received distance data: {data}")
                else:
                    print(f"Received malformed data: {data}")  # Handle unexpected data formats

        time.sleep(0.05)  # Match the Arduino delay (50ms)
except serial.SerialException as e:
    print(f"Error: {e}")
except KeyboardInterrupt:
    print("\nExiting...")
    ser.close()
