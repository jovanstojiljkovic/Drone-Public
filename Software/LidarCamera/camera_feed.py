import cv2

def main():
    # Initialize the camera (0 is the default camera index for most systems)
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
        
        # Display the resulting frame
        cv2.imshow('Video Feed', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
