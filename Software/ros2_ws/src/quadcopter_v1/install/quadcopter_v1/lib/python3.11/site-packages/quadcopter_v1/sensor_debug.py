import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Range

class DebugDistanceSensor(Node):
    def __init__(self):
        super().__init__('debug_distance_sensor')
        self.subscription = self.create_subscription(
            Range,
            '/quadcopter/right_distance_plugin/out',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        try:
            self.get_logger().info(f"Range: {msg.range}, Frame ID: {msg.header.frame_id}")
        except Exception as e:
            self.get_logger().error(f"Error decoding message: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = DebugDistanceSensor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
