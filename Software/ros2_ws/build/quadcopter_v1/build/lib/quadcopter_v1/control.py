import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from math import sin, cos

class PoseController(Node):
    def __init__(self):
        super().__init__('pose_controller')
        self.publisher = self.create_publisher(Pose, '/cmd_pose', 10)
        self.timer = self.create_timer(0.1, self.publish_pose)
        self.time_elapsed = 0.0

    def publish_pose(self):
        # Create a Pose message
        pose = Pose()
        self.time_elapsed += 0.1

        # Example motion: oscillate in x and rotate around z
        pose.position.x = 2.0 * sin(self.time_elapsed)  # Oscillate along x-axis
        pose.position.y = 0.0  # No movement along y-axis
        pose.position.z = 1.0 + 0.5 * sin(self.time_elapsed)  # Oscillate up and down
        pose.orientation.z = sin(self.time_elapsed / 2)  # Simple rotation
        pose.orientation.w = cos(self.time_elapsed / 2)

        # Publish the Pose message
        self.publisher.publish(pose)
        self.get_logger().info(f"Publishing Pose: {pose}")

def main(args=None):
    rclpy.init(args=args)
    node = PoseController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
