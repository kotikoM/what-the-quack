#!/usr/bin/env python3
from duckietown_msgs.msg import WheelsCmdStamped
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float64
from cv_bridge import CvBridge
import rospy
import time
import cv2
import os

from utils.sign_detection import SignDetector
from utils.lane_follower import LaneFollower


class CoordinatorNode(DTROS):
    road_signs = {
        20: "stop",
        24: "stop",
        26: "stop",
        31: "stop",
        32: "stop",
        33: "stop",
        96: "slow down",
        125: "parking"
    }

    def __init__(self):
        super(CoordinatorNode, self).__init__(node_name='coordinator_node', node_type=NodeType.VISUALIZATION)

        self.bridge = CvBridge()
        self.sign_detector = SignDetector()
        self.lane_follower = LaneFollower()
        self.sign_activated_time = 0
        self.sign_active_duration = 5
        self.current_sign = None

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

    def on_image_processing(self, msg):
        image = self.bridge.compressed_imgmsg_to_cv2(msg)
        result = self.sign_detector.detect_sign(image)

        cropped_frame = result["frame"]
        detections = result["detections"]
        left, right = self.lane_follower.follow_lane(image)
        cv2.imshow("right side feed", cropped_frame)
        current_time = time.time()

        # Check if current sign logic should expire
        if self.current_sign and (current_time - self.sign_activated_time > self.sign_active_duration):
            rospy.loginfo(f"Sign '{self.current_sign}' expired.")
            self.current_sign = None

        if self.current_sign:
            if self.current_sign == 'stop':
                left, right = 0, 0
            elif self.current_sign == 'slow down':
                left /= 2
                right /= 2
            elif self.current_sign == 'parking':
                left = 1.0
                right = -1.0
                rospy.loginfo("Executing parking logic...")
        else:
            for d in detections:
                if d.tag_id not in self.road_signs:
                    continue

                sign_name = self.road_signs[d.tag_id]
                self.current_sign = sign_name
                self.sign_activated_time = current_time
                rospy.loginfo(f"Activated sign logic for: {sign_name}")
                break

        # Actuate motors
        self.publish_left_motor(left)
        self.publish_right_motor(right)
        cv2.waitKey(1)

    def shutdown_hook(self):
        """Clean shutdown procedure"""
        rospy.loginfo("Shutting down CoordinatorNode...")
        self.shutting_down = True
        self.left_motor.publish(0.0)
        self.right_motor.publish(0.0)
        cv2.destroyAllWindows()

    def publish_left_motor(self, speed):
        self.left_motor.publish(Float64(speed))

    def publish_right_motor(self, speed):
        self.right_motor.publish(Float64(speed))


if __name__ == '__main__':
    node = CoordinatorNode()
    rospy.spin()
