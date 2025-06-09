#!/usr/bin/env python3
from duckietown_msgs.msg import WheelsCmdStamped, LEDPattern
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float64
from cv_bridge import CvBridge
import rospy
import cv2
import os


from utils.sign_detection import SignDetector


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

        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self.wheels_topic = f"/{self._vehicle_name}/wheels_driver_node/wheels_cmd"
        self.led_topic = f"/{self._vehicle_name}/led_emitter_node/led_pattern"

        self._window_name = "Duck Camera Feed"
        cv2.namedWindow(self._window_name, cv2.WINDOW_AUTOSIZE)

        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.on_image_processing)
        self._publisher = rospy.Publisher(self.wheels_topic, WheelsCmdStamped, queue_size=1)
        self.led_pub = rospy.Publisher(self.led_topic, LEDPattern, queue_size=1)

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
        image = self.bridge.compressed_imgmsg_to_cv2(msg)
        result = self.sign_detector.detect_sign(image)

        cropped_frame = result["frame"]
        detections = result["detections"]

        cv2.imshow("right side feed", cropped_frame)
        for d in detections:
            # rospy.loginfo(f"Detected tag ID: {d.tag_id}")

            if d.tag_id not in self.road_signs.keys():
                continue

            sign_name = self.road_signs.get(d.tag_id)
            if sign_name == 'stop':
                rospy.loginfo("Detected stop sign")
                self.set_led("RED")
            elif sign_name == 'slow down':
                rospy.loginfo("Detected slow down sign")
                self.set_led("YELLOW")
            elif sign_name == 'parking':
                rospy.loginfo("Detected parking sign")
                self.set_led("BLUE")


        cv2.waitKey(1)

    def set_led(self, pattern_name):
        pass
        # msg = LEDPattern()
        # msg.pattern = pattern_name
        # self.led_pub.publish(msg)

    def publish_left_motor(self, speed):
        self.left_motor.publish(Float64(speed))

    def publish_right_motor(self, speed):
        self.right_motor.publish(Float64(speed))


if __name__ == '__main__':
    node = CoordinatorNode()
    rospy.spin()
