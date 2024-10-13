import cv2 #opencv
import serial #reading serial terminal
import time

class CameraLidarApp:
    def __init__(self):
    #Initialize the camera:
         self.camera = cv2.VideoCapture(0)

         if not self.camera.isOpened():
            raise Exception("Could not start camera")
        
        #Set frame dimensions:
        self.frame_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

         # Set up the serial connection to Arduino 
        self.arduino = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)  #Change port accordingly
        time.sleep(2)  # Delay to let serial connection initialize
        print("Arduino Serial says: SERBUS")

    #Function to read lidar data from the serial of arduino
    def read_lidar(self):
       try:
            #Read a line of data from the serial connection
            if self.arduino.in_waiting > 0:
                data = self.arduino.readline().decode('utf-8').strip()
                return data
            else:
                return "No data"
        except Exception as e:
            print(f"Error reading from serial: {e}")
            return "Error"

    #Function for adding edge detetction using opencv:
    def edge_detect(self, frame):
        #Convert to Grayscale so that we can process it easier - Reduced model complexity:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #Apply Canny Edge detetction:
        edges = cv2.Canny(gray_frame, 100, 200)

        return edges
    #Main function
    def run(self):
        try:
            while True:
                #Capture frame-by-frame
                ret, frame = self.camera.read()
                if not ret:
                    print("E - Failed to grab frame")
                    break

                #Get LiDAR data from arduino:
                lidar_data = self.read_lidar()
                print(f"LiDAR readings: {lidar_data}")

                #process frame:
                processed_frame = self.edge_detect(frame)
                
                #show processed frame:
                cv2.imshow("Camera with Canny Edge Detection", processed_frame)
                
                #Exit everything when you click q
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            # Release camera and close all OpenCV windows
            self.camera.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    app = CameraLidarApp()
    app.run()
