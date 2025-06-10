#!/usr/bin/env python3
from duckietown_msgs.msg import WheelsCmdStamped
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float64, String
from cv_bridge import CvBridge
import rospy
import time
import cv2
import os

from utils.sign_detection import SignDetector


class CoordinatorNode(DTROS):
    road_signs = {
        20: "stop",
        24: "stop",
        25: "stop",
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

        self.current_sign = None
        self.sign_activated_time = None
        self.sign_reset_pending = False
        self.sign_active_duration = 5  # seconds
        self.freedom_published = False

        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"

        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.on_image_processing)
        self.sign_publisher = rospy.Publisher("sign_topic", String, queue_size=1)
        rospy.loginfo(f"CoordinatorNode initialized for vehicle: {self._vehicle_name}")

    def on_image_processing(self, msg):
        current_time = time.time()
        if self.current_sign and (current_time - self.sign_activated_time > self.sign_active_duration):
            rospy.loginfo(f"Sign '{self.current_sign}' expired.")
            self.current_sign = None
            if not self.freedom_published:
                self.sign_publisher.publish('freedom')
                rospy.loginfo("Published: freedom")
                self.freedom_published = True

        # Only look for new sign if no current one
        if self.current_sign is None:
            image = self.bridge.compressed_imgmsg_to_cv2(msg)
            result = self.sign_detector.detect_sign(image)

            cropped_frame = result["frame"]
            detections = result["detections"]
            cv2.imshow("right side feed", cropped_frame)

            for d in detections:
                if d.tag_id not in self.road_signs:
                    continue

                sign_name = self.road_signs[d.tag_id]
                self.current_sign = sign_name
                self.sign_activated_time = current_time
                self.sign_publisher.publish(sign_name)
                rospy.loginfo(f"Activated sign logic for: {sign_name}")
                self.freedom_published = False
                break

        cv2.waitKey(1)

if __name__ == '__main__':
    node = CoordinatorNode()
    rospy.spin()
