import cv2
import serial
import time
import threading

# Configuration for serial connection
serial_port = '/dev/ttyACM0'  # Update this to match your setup
baud_rate = 115200  # Match the baud rate of your Arduino

# Global variables for sensor data
sensor_data = {"angle": None, "distance": None}

def read_serial_data():
    """
    Reads data from the serial port and updates the global sensor_data dictionary.
    """
    try:
        ser = serial.Serial(serial_port, baud_rate)
        print(f"Connected to {serial_port} at {baud_rate} baud.")

        while True:
            if ser.in_waiting > 0:  # Check if data is available
                data = ser.readline().decode('utf-8', errors='ignore').strip()
                try:
                    # Parse angle data if possible
                    sensor_data["angle"] = float(data)
                except ValueError:
                    # If it's not angle data, check for distance (contains 'cm')
                    if "cm" in data:
                        sensor_data["distance"] = data
                    else:
                        print(f"Received malformed data: {data}")  # Handle unexpected data formats

            time.sleep(0.05)  # Match Arduino delay
    except serial.SerialException as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("\nExiting...")
        ser.close()

def main():
    # Start the serial data reading thread
    serial_thread = threading.Thread(target=read_serial_data, daemon=True)
    serial_thread.start()

    # Initialize the camera
    cap = cv2.VideoCapture(0)

    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    print("Press 'q' to exit the video feed.")
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # Check if frame is read correctly
        if not ret:
            print("Error: Unable to read from the camera.")
            break

        # Overlay sensor data on the frame
        angle = sensor_data["angle"]
        distance = sensor_data["distance"]

        overlay_text = []
        if angle is not None:
            overlay_text.append(f"Angle: {angle:.2f} degrees")
        if distance is not None:
            overlay_text.append(f"Distance: {distance}")

        for i, text in enumerate(overlay_text):
            cv2.putText(frame, text, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Video Feed with Sensor Data', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
