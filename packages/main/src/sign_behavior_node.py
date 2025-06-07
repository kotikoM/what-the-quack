#!/usr/bin/env python3

import rospy
from duckietown.dtros import DTROS, NodeType
from std_msgs.msg import Int8, Float64
import time

class SignBehaviorNode(DTROS):

    def __init__(self, node_name):
        super(SignBehaviorNode, self).__init__(
            node_name=node_name, node_type=NodeType.GENERIC)
        
        # Parameters
        self.vehicle_name = os.environ['VEHICLE_NAME']
        self.current_behavior = None
        self.behavior_start_time = None
        self.behavior_duration = 0
        self.original_left = 0
        self.original_right = 0
        
        # Subscribers
        self.sub_sign = rospy.Subscriber(
            "detected_sign", Int8, self.sign_callback, queue_size=1)
        self.sub_left = rospy.Subscriber(
            "left_motor", Float64, self.left_callback, queue_size=1)
        self.sub_right = rospy.Subscriber(
            "right_motor", Float64, self.right_callback, queue_size=1)
            
        # Publishers (will override camera node commands)
        self.pub_left = rospy.Publisher(
            "left_motor_adj", Float64, queue_size=1)
        self.pub_right = rospy.Publisher(
            "right_motor_adj", Float64, queue_size=1)

    def sign_callback(self, msg):
        sign_id = msg.data
        rospy.loginfo(f"Detected sign: {sign_id}")
        
        # Define behaviors for different signs
        if sign_id == 1:  # Stop sign
            self.current_behavior = "stop"
            self.behavior_duration = 3.0  # Stop for 3 seconds
            
        elif sign_id == 2:  # Parking sign
            self.current_behavior = "park"
            self.behavior_duration = 5.0  # Park for 5 seconds
            
        elif sign_id == 3:  # Slow down sign
            self.current_behavior = "slow"
            self.behavior_duration = 5.0  # Slow down for 5 seconds
            
        self.behavior_start_time = time.time()

    def left_callback(self, msg):
        self.original_left = msg.data
        self.apply_behavior()

    def right_callback(self, msg):
        self.original_right = msg.data
        self.apply_behavior()

    def apply_behavior(self):
        if self.current_behavior is None:
            # No active behavior - pass through commands
            self.pub_left.publish(self.original_left)
            self.pub_right.publish(self.original_right)
            return
            
        elapsed = time.time() - self.behavior_start_time
        
        if elapsed > self.behavior_duration:
            # Behavior completed
            self.current_behavior = None
            self.pub_left.publish(self.original_left)
            self.pub_right.publish(self.original_right)
            return
            
        # Apply current behavior
        if self.current_behavior == "stop":
            self.pub_left.publish(0.0)
            self.pub_right.publish(0.0)
            
        elif self.current_behavior == "park":
            # For parking, stop completely
            self.pub_left.publish(0.0)
            self.pub_right.publish(0.0)
            
        elif self.current_behavior == "slow":
            # Reduce speed to 50% of original
            self.pub_left.publish(self.original_left * 0.5)
            self.pub_right.publish(self.original_right * 0.5)

if __name__ == '__main__':
    node = SignBehaviorNode(node_name='sign_behavior_node')
    rospy.spin()