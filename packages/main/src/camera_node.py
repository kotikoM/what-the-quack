#!/usr/bin/env python3
import os
import rospy
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import WheelsCmdStamped
import numpy as np
import cv2
from cv_bridge import CvBridge
from std_msgs.msg import Float64
from collections import deque

# Tunable parameters
BASE_SPEED = 0.25  # Base forward speed
CURVE_SPEED = 0.20  # Reduced speed for curves
P_GAIN = 0.4  # Proportional gain
D_GAIN = 0.2  # Derivative gain - helps with curves
MAX_STEER = 0.5  # Maximum steering adjustment
SMOOTHING_STRAIGHT = 3  # Smoothing for straight roads
SMOOTHING_CURVE = 2  # Less smoothing for curves


class CameraReaderNode(DTROS):

    def __init__(self, node_name):
        super(CameraReaderNode, self).__init__(
            node_name=node_name, node_type=NodeType.VISUALIZATION)

        # Initialize parameters
        self.base_speed = BASE_SPEED
        self.curve_speed = CURVE_SPEED
        self.p_gain = P_GAIN
        self.d_gain = D_GAIN
        self.max_steer = MAX_STEER

        # Error tracking for derivative control
        self.prev_error = 0
        self.left_motor_history = deque(maxlen=SMOOTHING_STRAIGHT)
        self.right_motor_history = deque(maxlen=SMOOTHING_STRAIGHT)

        # Setup ROS nodes
        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        wheels_topic = f"/{self._vehicle_name}/wheels_driver_node/wheels_cmd"

        # Create multiple windows for debugging
        self._main_window = "Lane Detection - Main View"
        self._yellow_window = "Yellow Mask"
        self._white_window = "White Mask"
        self._edges_window = "Detected Edges"
        self.bridge = CvBridge()
        cv2.namedWindow(self._main_window, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(self._yellow_window, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(self._white_window, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(self._edges_window, cv2.WINDOW_AUTOSIZE)

        self.sub = rospy.Subscriber(
            self._camera_topic, CompressedImage, self.callback)
        self._publisher = rospy.Publisher(
            wheels_topic, WheelsCmdStamped, queue_size=1)

        self.left_motor = rospy.Publisher("left_motor", Float64, queue_size=1)
        self.right_motor = rospy.Publisher("right_motor", Float64, queue_size=1)

        self.shutting_down = False
        rospy.on_shutdown(self.shutdown_hook)

    def shutdown_hook(self):
        self.shutting_down = True
        self.left_motor.publish(0)
        self.right_motor.publish(0)
        cv2.destroyAllWindows()

    def smooth_motor_value(self, value, history_buffer):
        """Apply smoothing to motor values"""
        history_buffer.append(value)
        return sum(history_buffer) / len(history_buffer)

    def create_debug_displays(self, mask_yellow, mask_white, error, steering, is_curve,
                              curve_direction, left_motor, right_motor):
        """Create separate debug displays for yellow mask, white mask, and detected edges"""
        # Create edge detection from combined masks
        combined_mask = cv2.bitwise_or(mask_yellow, mask_white)
        edges = cv2.Canny(combined_mask, 50, 150)

        # Convert single channel masks to 3-channel for display
        yellow_display = cv2.cvtColor(mask_yellow, cv2.COLOR_GRAY2BGR)
        white_display = cv2.cvtColor(mask_white, cv2.COLOR_GRAY2BGR)
        edges_display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # Add debug info to each display
        info_text = [
            f"Error: {error:.3f} | Steering: {steering:.3f}",
            f"Curve: {is_curve} | Direction: {curve_direction}",
            f"Motors - L: {left_motor:.3f} R: {right_motor:.3f}",
            f"Yellow pixels: {np.count_nonzero(mask_yellow)}",
            f"White pixels: {np.count_nonzero(mask_white)}"
        ]

        # Add info to yellow display
        for i, text in enumerate(info_text[:3]):
            cv2.putText(yellow_display, text, (10, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(yellow_display, info_text[3], (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Add info to white display
        for i, text in enumerate(info_text[:3]):
            cv2.putText(white_display, text, (10, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(white_display, info_text[4], (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Add info to edges display
        for i, text in enumerate(info_text[:3]):
            cv2.putText(edges_display, text, (10, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return yellow_display, white_display, edges_display

    def callback(self, msg):
        if self.shutting_down:
            return

        # Process image
        self.image = self.bridge.compressed_imgmsg_to_cv2(msg)
        original_image = self.image.copy()
        self.image = cv2.bilateralFilter(self.image, 9, 75, 75)  # Less aggressive filtering

        # Create a copy for visualization with enhanced overlays
        vis_image = self.image.copy()

        # Define regions of interest - near field and far field
        h, w = self.image.shape[:2]
        near_field = self.image[int(h * 0.6):, :]  # Bottom 40% of image
        far_field = self.image[int(h * 0.4):int(h * 0.6), :]  # Middle of image

        # Draw ROI boundaries on visualization
        roi_start_y = int(h * 0.55)
        cv2.line(vis_image, (0, roi_start_y), (w, roi_start_y), (255, 0, 255), 2)
        cv2.putText(vis_image, "ROI Start", (10, roi_start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        # Color space conversions
        luv = cv2.cvtColor(self.image, cv2.COLOR_BGR2LUV)
        hls = cv2.cvtColor(self.image, cv2.COLOR_BGR2HLS)

        # Yellow line detection (usually left boundary)
        lb_yellow = np.array([10, 85, 160])  # Slightly adjusted values
        ub_yellow = np.array([255, 255, 255])
        mask_yellow = cv2.inRange(luv, lb_yellow, ub_yellow)
        # Apply region of interest - focus on bottom part of image
        mask_yellow[:int(h * 0.55), :] = 0

        # White line detection (usually right boundary)
        lb_white = np.array([0, 144, 0])  # Higher lightness threshold
        ub_white = np.array([168, 255, 36])
        mask_white = cv2.inRange(hls, lb_white, ub_white)
        # Apply region of interest - focus on bottom part of image
        mask_white[:int(h * 0.55), :] = 0

        # Apply morphological operations to clean up masks
        kernel = np.ones((5, 5), np.uint8)
        mask_yellow = cv2.dilate(mask_yellow, kernel, iterations=1)
        mask_white = cv2.dilate(mask_white, kernel, iterations=1)

        # Find contours to detect line shape
        yellow_contours, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        white_contours, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Enhanced contour visualization with area filtering
        filtered_yellow_contours = [c for c in yellow_contours if cv2.contourArea(c) > 100]
        filtered_white_contours = [c for c in white_contours if cv2.contourArea(c) > 100]

        # Draw all contours with different colors and add area info
        for i, contour in enumerate(filtered_yellow_contours):
            area = cv2.contourArea(contour)
            cv2.drawContours(vis_image, [contour], -1, (0, 255, 255), 3)
            # Get contour center for labeling
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(vis_image, f"Y{i}:{int(area)}", (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255),
                            1)

        for i, contour in enumerate(filtered_white_contours):
            area = cv2.contourArea(contour)
            cv2.drawContours(vis_image, [contour], -1, (255, 255, 255), 3)
            # Get contour center for labeling
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(vis_image, f"W{i}:{int(area)}", (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (255, 255, 255), 1)

        # Initialize variables for line detection
        left_line_detected = len(filtered_yellow_contours) > 0
        right_line_detected = len(filtered_white_contours) > 0

        # Split the image into slices to detect curve
        num_slices = 3
        slice_height = int(h * 0.35 / num_slices)
        start_y = int(h * 0.55)

        yellow_x_points = []
        white_x_points = []
        y_points = []

        # Draw slice boundaries
        for i in range(num_slices + 1):
            y = start_y + i * slice_height
            cv2.line(vis_image, (0, y), (w, y), (128, 128, 128), 1)

        for i in range(num_slices):
            y = start_y + i * slice_height + slice_height // 2
            y_points.append(y)

            # Draw slice center line
            cv2.line(vis_image, (0, y), (w, y), (0, 128, 255), 1)

            # Find yellow line x-position in this slice
            slice_yellow = mask_yellow[y - 5:y + 5, :]
            yellow_indices = np.where(slice_yellow > 0)[1]
            if len(yellow_indices) > 0:
                yellow_x = int(np.mean(yellow_indices))
                yellow_x_points.append(yellow_x)
                cv2.circle(vis_image, (yellow_x, y), 8, (0, 255, 255), -1)
                cv2.circle(vis_image, (yellow_x, y), 12, (0, 0, 0), 2)
                cv2.putText(vis_image, f"Y{i}", (yellow_x + 15, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            # Find white line x-position in this slice
            slice_white = mask_white[y - 5:y + 5, :]
            white_indices = np.where(slice_white > 0)[1]
            if len(white_indices) > 0:
                white_x = int(np.mean(white_indices))
                white_x_points.append(white_x)
                cv2.circle(vis_image, (white_x, y), 8, (255, 255, 255), -1)
                cv2.circle(vis_image, (white_x, y), 12, (0, 0, 0), 2)
                cv2.putText(vis_image, f"W{i}", (white_x + 15, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255),
                            1)

        # Draw lines connecting the detected points
        if len(yellow_x_points) > 1:
            yellow_pts = np.array([[yellow_x_points[i], y_points[i]] for i in range(len(yellow_x_points))], np.int32)
            cv2.polylines(vis_image, [yellow_pts], False, (0, 200, 200), 3)

        if len(white_x_points) > 1:
            white_pts = np.array([[white_x_points[i], y_points[i]] for i in range(len(white_x_points))], np.int32)
            cv2.polylines(vis_image, [white_pts], False, (200, 200, 200), 3)

        # Detect if we're in a curve by checking the difference in line positions
        is_curve = False
        curve_direction = 0

        if len(yellow_x_points) >= 2:
            yellow_diff = yellow_x_points[-1] - yellow_x_points[0]
            if abs(yellow_diff) > 20:  # Threshold for curve detection
                is_curve = True
                curve_direction += np.sign(yellow_diff)

        if len(white_x_points) >= 2:
            white_diff = white_x_points[-1] - white_x_points[0]
            if abs(white_diff) > 20:  # Threshold for curve detection
                is_curve = True
                curve_direction += np.sign(white_diff)

        # Determine center position and error
        center_position = None
        if left_line_detected and right_line_detected and len(yellow_x_points) > 0 and len(white_x_points) > 0:
            # Both lines visible - use both for guidance
            center_position = (yellow_x_points[-1] + white_x_points[-1]) / 2
            ideal_center = w / 2
            error = ideal_center - center_position
            cv2.putText(vis_image, "BOTH LINES", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        elif left_line_detected and len(yellow_x_points) > 0:
            # Only left (yellow) line visible - estimate center
            center_position = yellow_x_points[-1] + 160
            error = w / 2 - center_position
            cv2.putText(vis_image, "YELLOW ONLY", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        elif right_line_detected and len(white_x_points) > 0:
            # Only right (white) line visible - estimate center
            center_position = white_x_points[-1] - 160
            error = w / 2 - center_position
            cv2.putText(vis_image, "WHITE ONLY", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            # No lines visible - maintain last direction but be cautious
            error = self.prev_error
            cv2.putText(vis_image, "NO LINES", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Draw center position and target
        if center_position is not None:
            cv2.circle(vis_image, (int(center_position), int(h * 0.8)), 10, (255, 0, 0), -1)
            cv2.putText(vis_image, "CENTER", (int(center_position) + 15, int(h * 0.8)), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 0, 0), 1)

        # Draw ideal center
        cv2.circle(vis_image, (int(w / 2), int(h * 0.8)), 10, (0, 255, 0), -1)
        cv2.putText(vis_image, "TARGET", (int(w / 2) + 15, int(h * 0.8)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Draw error arrow
        if center_position is not None:
            arrow_start = (int(center_position), int(h * 0.85))
            arrow_end = (int(w / 2), int(h * 0.85))
            cv2.arrowedLine(vis_image, arrow_start, arrow_end, (255, 0, 255), 3)

        # Normalize error to range [-1, 1]
        error = np.clip(error / (w / 2), -1, 1)

        # Calculate derivative of error for D-term
        error_diff = error - self.prev_error
        self.prev_error = error

        # PD control
        steering = self.p_gain * error + self.d_gain * error_diff
        steering = np.clip(steering, -self.max_steer, self.max_steer)

        # Adjust speeds based on curve detection
        current_speed = self.curve_speed if is_curve else self.base_speed

        # Calculate motor values
        left_motor = current_speed - steering
        right_motor = current_speed + steering

        # Special case: if in a tight curve, help by differential steering
        if is_curve and abs(steering) > 0.3:
            # Boost the inside wheel in curves
            if steering > 0:  # Turning left
                right_motor *= 1.3
            else:  # Turning right
                left_motor *= 1.3

        # Recovery behavior if no lines detected
        line_pixels = np.count_nonzero(mask_yellow) + np.count_nonzero(mask_white)
        if line_pixels < 500:  # Very few line pixels detected
            # Back up and turn to recover
            left_motor = -0.2
            right_motor = -0.3
            cv2.putText(vis_image, "RECOVERY MODE", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Determine smoothing amount based on curve detection
        smoothing_amount = SMOOTHING_CURVE if is_curve else SMOOTHING_STRAIGHT
        self.left_motor_history = deque(self.left_motor_history, maxlen=smoothing_amount)
        self.right_motor_history = deque(self.right_motor_history, maxlen=smoothing_amount)

        # Apply smoothing
        left_motor = self.smooth_motor_value(left_motor, self.left_motor_history)
        right_motor = self.smooth_motor_value(right_motor, self.right_motor_history)

        # Safety bounds
        left_motor = np.clip(left_motor, -1.0, 1.0)
        right_motor = np.clip(right_motor, -1.0, 1.0)

        # Publish motor commands
        if not self.shutting_down:
            self.left_motor.publish(left_motor)
            self.right_motor.publish(right_motor)

        # Add enhanced status display to main visualization
        status_texts = [
            f"Curve: {is_curve} | Direction: {curve_direction}",
            f"Error: {error:.3f} | Steering: {steering:.3f}",
            f"Speed: {current_speed:.2f} | Motors: L={left_motor:.3f} R={right_motor:.3f}",
            f"Line pixels: Y={np.count_nonzero(mask_yellow)} W={np.count_nonzero(mask_white)}",
            f"Contours: Y={len(filtered_yellow_contours)} W={len(filtered_white_contours)}"
        ]

        for i, text in enumerate(status_texts):
            cv2.putText(vis_image, text, (10, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Create separate debug displays
        yellow_display, white_display, edges_display = self.create_debug_displays(
            mask_yellow, mask_white, error, steering, is_curve, curve_direction, left_motor, right_motor
        )

        # Display all windows separately
        cv2.imshow(self._main_window, vis_image)
        cv2.imshow(self._yellow_window, yellow_display)
        cv2.imshow(self._white_window, white_display)
        cv2.imshow(self._edges_window, edges_display)
        cv2.waitKey(1)


if __name__ == '__main__':
    node = CameraReaderNode(node_name='camera_reader_node')
    rospy.spin()
