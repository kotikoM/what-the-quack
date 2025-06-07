#!/usr/bin/env python3

import os
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelsCmdStamped
from std_msgs.msg import Float64
from cv_bridge import CvBridge


class WheelControlNode(DTROS):

    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(WheelControlNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        VEHICLE_NAME_KEY = 'VEHICLE_NAME'
        self.vehicle_name = os.environ[VEHICLE_NAME_KEY]
        wheels_topic = f"/{self.vehicle_name}/wheels_driver_node/wheels_cmd"

        self._vel_left = 0
        self._vel_right = 0

        # construct publisher
        self._publisher = rospy.Publisher(wheels_topic, WheelsCmdStamped, queue_size=1)

        self.left_motor = rospy.Subscriber("left_motor", Float64, self.callback_left)
        self.right_motor = rospy.Subscriber("right_motor", Float64, self.callback_right)
        self.bridge = CvBridge()

    def callback_left(self, msg):
        self._vel_left = msg.data

    def callback_right(self, msg):
        self._vel_right = msg.data

    def run(self):
        # publish 10 messages every second (10 Hz)
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            message = WheelsCmdStamped(vel_left=self._vel_left, vel_right=self._vel_right)

            self._publisher.publish(message)
            rate.sleep()

    def on_shutdown(self):
        stop = WheelsCmdStamped(vel_left=0, vel_right=0)
        self._publisher.publish(stop)


if __name__ == '__main__':
    node = WheelControlNode(node_name='wheel_control_node')
    rospy.on_shutdown(node.on_shutdown)
    node.run()
    rospy.spin()
