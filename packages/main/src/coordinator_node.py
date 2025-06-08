#!/usr/bin/env python3
import os
import rospy
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import WheelsCmdStamped
from std_msgs.msg import Float64
from cv_bridge import CvBridge
import cv2

from utils.sign_detection import SignDetector


class CoordinatorNode(DTROS):
    def __init__(self):
        super(CoordinatorNode, self).__init__(node_name='coordinator_node', node_type=NodeType.VISUALIZATION)

        self.bridge = CvBridge()
        self.detector = SignDetector()

        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self.wheels_topic = f"/{self._vehicle_name}/wheels_driver_node/wheels_cmd"

        self._window_name = "Duck Camera Feed"
        cv2.namedWindow(self._window_name, cv2.WINDOW_AUTOSIZE)

        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.on_image_processing)
        self._publisher = rospy.Publisher(self.wheels_topic, WheelsCmdStamped, queue_size=1)

        self.left_motor = rospy.Publisher("left_motor", Float64, queue_size=1)
        self.right_motor = rospy.Publisher("right_motor", Float64, queue_size=1)

        self.shutting_down = False
        rospy.on_shutdown(self.shutdown_hook)
        rospy.loginfo(f"CoordinatorNode initialized for vehicle: {self._vehicle_name}")

    def shutdown_hook(self):
        """Clean shutdown procedure"""
        rospy.loginfo("Shutting down CoordinatorNode...")
        self.shutting_down = True
        self.left_motor.publish(0.0)
        self.right_motor.publish(0.0)
        cv2.destroyAllWindows()

    def on_image_processing(self, msg):
        """Process incoming compressed image messages"""
        # Convert compressed image to OpenCV format using cv_bridge
        image = self.bridge.compressed_imgmsg_to_cv2(msg)
        result = self.detector.process_image(image)

        frame1 = result["contours_frame"]
        frame2 = result["squares_frame"]
        warped_images = result["warped_images"]

        cv2.imshow("contours", frame1)
        cv2.imshow("squares", frame2)

        for i, warped in enumerate(warped_images):
            cv2.imshow(f"warped_{i}", warped)

        cv2.waitKey(1)

    def publish_left_motor(self, speed):
        self.left_motor.publish(Float64(speed))

    def publish_right_motor(self, speed):
        self.right_motor.publish(Float64(speed))


if __name__ == '__main__':
    node = CoordinatorNode()
    rospy.spin()
