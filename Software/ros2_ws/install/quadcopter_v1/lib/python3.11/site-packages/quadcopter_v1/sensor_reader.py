import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import cv2  # OpenCV
import sensor_msgs_py.point_cloud2 as pc2  # For reading PointCloud2 data

class SensorReader(Node):
    def __init__(self):
        super().__init__('sensor_reader')

        self.bridge = CvBridge()  # Bridge for converting ROS images to OpenCV format

        # Subscribing to camera topic
        self.camera_sub = self.create_subscription(
            Image,
            '/quadcopter/camera/image_raw',
            self.camera_callback,
            10
        )

        # Subscribing to right distance sensor topic (PointCloud2)
        self.right_sensor_sub = self.create_subscription(
            PointCloud2,
            '/quadcopter/right_distance_plugin/out',
            self.right_distance_callback,
            10
        )

        # Subscribing to left distance sensor topic (PointCloud2)
        self.left_sensor_sub = self.create_subscription(
            PointCloud2,
            '/quadcopter/left_distance_plugin/out',
            self.left_distance_callback,
            10
        )

    def camera_callback(self, msg):
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Display the image
            cv2.imshow("Quadcopter Camera Feed", cv_image)
            cv2.waitKey(1)  # Wait for 1ms to allow OpenCV to refresh
        except Exception as e:
            self.get_logger().error(f"Failed to process image: {e}")

    def process_pointcloud(self, msg):
        """Extract the closest range from PointCloud2 data."""
        try:
            # Read points from PointCloud2 message
            points = list(pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True))

            # Calculate distances and return the closest one
            closest_distance = min((p[0]**2 + p[1]**2 + p[2]**2)**0.5 for p in points)
            return closest_distance
        except Exception as e:
            self.get_logger().error(f"Error processing PointCloud2 message: {e}")
            return None

    def right_distance_callback(self, msg):
        # Process PointCloud2 and log the closest distance
        distance = self.process_pointcloud(msg)
        if distance is not None:
            self.get_logger().info(f"Right distance sensor: {distance:.2f} meters")

    def left_distance_callback(self, msg):
        # Process PointCloud2 and log the closest distance
        distance = self.process_pointcloud(msg)
        if distance is not None:
            self.get_logger().info(f"Left distance sensor: {distance:.2f} meters")


def main(args=None):
    rclpy.init(args=args)
    sensor_reader = SensorReader()

    print("Script started...")

    try:
        rclpy.spin(sensor_reader)
    except KeyboardInterrupt:
        pass
    finally:
        sensor_reader.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()  # Clean up OpenCV windows


if __name__ == '__main__':
    main()
