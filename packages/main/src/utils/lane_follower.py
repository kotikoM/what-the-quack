from collections import deque
import numpy as np
import time
import cv2


BASE_SPEED = 0.18  # Reduced base speed for better control
CURVE_SPEED = 0.16  # Further reduced speed for curves
P_GAIN = 0.35  # Reduced proportional gain for smoother response
D_GAIN = 0.25  # Increased derivative gain for better curve handling
MAX_STEER = 0.2  # Reduced maximum steering adjustment
SMOOTHING_STRAIGHT = 5  # Increased smoothing for straight roads
SMOOTHING_CURVE = 3  # Moderate smoothing for curves
TURN_SLOWDOWN_FACTOR = 0.7  # Factor to slow down during turns

class LaneFollower:
    def __init__(self):
        # Initialize parameters
        self.base_speed = BASE_SPEED
        self.curve_speed = CURVE_SPEED
        self.p_gain = P_GAIN
        self.d_gain = D_GAIN
        self.max_steer = MAX_STEER

        # Lane following state tracking
        self.start_time = time.time()

        # Error tracking with more smoothing
        self.prev_error = 0
        self.error_history = deque(maxlen=5)  # Additional error smoothing
        self.left_motor_history = deque(maxlen=SMOOTHING_STRAIGHT)
        self.right_motor_history = deque(maxlen=SMOOTHING_STRAIGHT)

        # Line tracking for better curve handling
        self.yellow_centroid_history = deque(maxlen=5)
        self.white_centroid_history = deque(maxlen=5)
        self.lane_width_history = deque(maxlen=10)
        self._window = "camera-reader"
        self._debug_window = "debug-masks"
        cv2.namedWindow(self._window, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(self._debug_window, cv2.WINDOW_AUTOSIZE)


    def follow_lane(self, frame):
        pass