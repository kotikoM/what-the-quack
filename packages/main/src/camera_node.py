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

# Tunable parameters - Adjusted for smoother turning
BASE_SPEED = 0.18          # Reduced base speed for better control
CURVE_SPEED = 0.16         # Further reduced speed for curves
P_GAIN = 0.35              # Reduced proportional gain for smoother response
D_GAIN = 0.25              # Increased derivative gain for better curve handling
MAX_STEER = 0.2            # Reduced maximum steering adjustment
SMOOTHING_STRAIGHT = 5     # Increased smoothing for straight roads
SMOOTHING_CURVE = 3        # Moderate smoothing for curves
TURN_SLOWDOWN_FACTOR = 0.7 # Factor to slow down during turns

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
        
        # Error tracking with more smoothing
        self.prev_error = 0
        self.error_history = deque(maxlen=5)  # Additional error smoothing
        self.left_motor_history = deque(maxlen=SMOOTHING_STRAIGHT)
        self.right_motor_history = deque(maxlen=SMOOTHING_STRAIGHT)
        
        # Setup ROS nodes
        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        wheels_topic = f"/{self._vehicle_name}/wheels_driver_node/wheels_cmd"
        self._window = "camera-reader"
        self._debug_window = "debug-masks"
        self.bridge = CvBridge()
        cv2.namedWindow(self._window, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(self._debug_window, cv2.WINDOW_AUTOSIZE)

        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)
        self._publisher = rospy.Publisher(wheels_topic, WheelsCmdStamped, queue_size=1)

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
        """Apply smoothing to motor values with weighted average"""
        history_buffer.append(value)
        # Use weighted average (more weight to recent values)
        weights = np.linspace(1, 2, len(history_buffer))
        return np.average(history_buffer, weights=weights)

    def callback(self, msg):
        if self.shutting_down:
            return

        # Process image with less aggressive filtering
        self.image = self.bridge.compressed_imgmsg_to_cv2(msg)
        self.image = cv2.GaussianBlur(self.image, (5, 5), 0)  # Lighter blur
        
        # Create a copy for visualization
        vis_image = self.image.copy()
        
        # Define regions of interest
        h, w = self.image.shape[:2]
        roi_top = int(h*0.6)  # ROI starts at 60% down from top
        roi_bottom = self.image[roi_top:, :]  # Bottom 40% of image
        
        # DEBUG: Draw ROI rectangle on visualization
        cv2.rectangle(vis_image, (0, roi_top), (w-1, h-1), (255, 0, 0), 2)
        cv2.putText(vis_image, "ROI", (10, roi_top-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Color space conversions
        luv = cv2.cvtColor(roi_bottom, cv2.COLOR_BGR2LUV)
        hls = cv2.cvtColor(roi_bottom, cv2.COLOR_BGR2HLS)

        # Yellow line detection (left boundary)
        lb_yellow = np.array([10, 85, 160])
        ub_yellow = np.array([255, 255, 255])
        mask_yellow = cv2.inRange(luv, lb_yellow, ub_yellow)
        
        # White line detection (right boundary)
        lb_white = np.array([0, 180, 0])  # Higher lightness threshold
        ub_white = np.array([172, 255, 255])
        mask_white = cv2.inRange(hls, lb_white, ub_white)
        
        # Apply morphological operations
        kernel = np.ones((3, 3), np.uint8)  # Smaller kernel
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel)
        mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel)
        
        # Filter out white lines on the left side of the camera
        # Only keep white pixels on the right half of the image
        left_half_boundary = w // 2
        mask_white[:, :left_half_boundary] = 0  # Zero out left half
        
        # DEBUG: Create debug visualization showing all masks and detected pixels
        debug_image = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Show ROI area
        debug_image[roi_top:, :] = [50, 50, 50]  # Dark gray for ROI
        
        # Show the left-side exclusion zone for white lines
        cv2.line(debug_image, (left_half_boundary, roi_top), (left_half_boundary, h), (255, 0, 255), 2)
        cv2.putText(debug_image, "White line exclusion", (left_half_boundary + 5, roi_top + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        
        # Overlay yellow mask pixels in yellow
        yellow_pixels = np.where(mask_yellow > 0)
        if len(yellow_pixels[0]) > 0:
            debug_image[yellow_pixels[0] + roi_top, yellow_pixels[1]] = [0, 255, 255]  # Yellow
            
        # Overlay white mask pixels in white (only from right side now)
        white_pixels = np.where(mask_white > 0)
        if len(white_pixels[0]) > 0:
            debug_image[white_pixels[0] + roi_top, white_pixels[1]] = [255, 255, 255]  # White
        
        # Count pixels for debugging
        yellow_pixel_count = np.count_nonzero(mask_yellow)
        white_pixel_count = np.count_nonzero(mask_white)
        total_line_pixels = yellow_pixel_count + white_pixel_count
        
        # Find line centroids
        yellow_centroid = self.find_line_centroid(mask_yellow)
        white_centroid = self.find_line_centroid(mask_white)
        
        # DEBUG: Mark centroids on both images
        if yellow_centroid is not None:
            centroid_y_global = yellow_centroid[1] + roi_top
            cv2.circle(vis_image, (yellow_centroid[0], centroid_y_global), 8, (0, 255, 255), -1)
            cv2.circle(debug_image, (yellow_centroid[0], centroid_y_global), 8, (0, 200, 200), -1)
            cv2.putText(debug_image, "Y", (yellow_centroid[0]-10, centroid_y_global-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
        if white_centroid is not None:
            centroid_y_global = white_centroid[1] + roi_top
            cv2.circle(vis_image, (white_centroid[0], centroid_y_global), 8, (255, 255, 255), -1)
            cv2.circle(debug_image, (white_centroid[0], centroid_y_global), 8, (200, 200, 200), -1)
            cv2.putText(debug_image, "W", (white_centroid[0]-10, centroid_y_global-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Determine center position and error
        if yellow_centroid is not None and white_centroid is not None:
            # Both lines visible
            center_position = (yellow_centroid[0] + white_centroid[0]) / 2
            ideal_center = w / 2
            error = ideal_center - center_position
            line_spacing = abs(yellow_centroid[0] - white_centroid[0])
            
            # DEBUG: Draw center line and ideal center
            cv2.line(vis_image, (int(center_position), roi_top), 
                    (int(center_position), h), (0, 255, 0), 2)
            cv2.line(debug_image, (int(center_position), roi_top), 
                    (int(center_position), h), (0, 255, 0), 2)
            cv2.line(vis_image, (int(ideal_center), roi_top), 
                    (int(ideal_center), h), (255, 0, 0), 1)
            cv2.line(debug_image, (int(ideal_center), roi_top), 
                    (int(ideal_center), h), (255, 0, 0), 1)
                    
        elif yellow_centroid is not None:
            # Only left (yellow) line visible - estimate lane center
            # Assume standard lane width of ~200 pixels and position robot accordingly
            estimated_lane_width = 200
            estimated_center = yellow_centroid[0] + (estimated_lane_width / 2)
            ideal_center = w / 2
            error = ideal_center - estimated_center
            line_spacing = None
            
            # DEBUG: Draw estimated center and yellow line reference
            cv2.line(vis_image, (int(estimated_center), roi_top), 
                    (int(estimated_center), h), (0, 255, 0), 2)
            cv2.line(debug_image, (int(estimated_center), roi_top), 
                    (int(estimated_center), h), (0, 255, 0), 2)
            cv2.line(vis_image, (int(ideal_center), roi_top), 
                    (int(ideal_center), h), (255, 0, 0), 1)
            cv2.line(debug_image, (int(ideal_center), roi_top), 
                    (int(ideal_center), h), (255, 0, 0), 1)
            # Show estimated lane boundary
            cv2.line(vis_image, (int(yellow_centroid[0] + estimated_lane_width), roi_top), 
                    (int(yellow_centroid[0] + estimated_lane_width), h), (0, 150, 150), 1)
            cv2.line(debug_image, (int(yellow_centroid[0] + estimated_lane_width), roi_top), 
                    (int(yellow_centroid[0] + estimated_lane_width), h), (0, 150, 150), 1)
                    
        elif white_centroid is not None:
            # Only right (white) line visible - estimate lane center
            estimated_lane_width = 200
            estimated_center = white_centroid[0] - (estimated_lane_width / 2)
            ideal_center = w / 2
            error = ideal_center - estimated_center
            line_spacing = None
            
            # DEBUG: Draw estimated center and white line reference  
            cv2.line(vis_image, (int(estimated_center), roi_top), 
                    (int(estimated_center), h), (0, 255, 0), 2)
            cv2.line(debug_image, (int(estimated_center), roi_top), 
                    (int(estimated_center), h), (0, 255, 0), 2)
            cv2.line(vis_image, (int(ideal_center), roi_top), 
                    (int(ideal_center), h), (255, 0, 0), 1)
            cv2.line(debug_image, (int(ideal_center), roi_top), 
                    (int(ideal_center), h), (255, 0, 0), 1)
            # Show estimated lane boundary
            cv2.line(vis_image, (int(white_centroid[0] - estimated_lane_width), roi_top), 
                    (int(white_centroid[0] - estimated_lane_width), h), (150, 150, 0), 1)
            cv2.line(debug_image, (int(white_centroid[0] - estimated_lane_width), roi_top), 
                    (int(white_centroid[0] - estimated_lane_width), h), (150, 150, 0), 1)
        else:
            # No lines visible - use previous error with decay
            error = self.prev_error * 0.8
            line_spacing = None
        
        # Normalize error to range [-1, 1]
        error = np.clip(error / (w/2), -1, 1)
        self.error_history.append(error)
        smoothed_error = np.mean(self.error_history)  # Additional error smoothing
        
        # Calculate derivative of error for D-term
        error_diff = smoothed_error - self.prev_error
        self.prev_error = smoothed_error
        
        # PD control with smoother response
        steering = self.p_gain * smoothed_error + self.d_gain * error_diff
        steering = np.clip(steering, -self.max_steer, self.max_steer)
        
        # Detect curves based on line spacing and error
        is_curve = False
        if line_spacing is not None and line_spacing < 200:  # Lines getting closer indicates curve
            is_curve = True
        
        # Adjust speeds based on curve detection and steering amount
        if is_curve:
            current_speed = self.curve_speed
            # Slow down more when steering harder
            speed_factor = 1 - (abs(steering) * TURN_SLOWDOWN_FACTOR)
            current_speed *= max(0.5, speed_factor)  # Don't go below 50% speed
        else:
            current_speed = self.base_speed
        
        # Calculate motor values with smoother transitions
        left_motor = current_speed - steering
        right_motor = current_speed + steering
        
        # Recovery behavior if no lines detected
        if total_line_pixels < 300:  # Very few line pixels detected
            # Slow down and gently turn to find lines
            left_motor = current_speed * 0.5 + 0.1
            right_motor = current_speed * 0.5 - 0.1
        
        # Apply smoothing based on curve detection
        smoothing_amount = SMOOTHING_CURVE if is_curve else SMOOTHING_STRAIGHT
        self.left_motor_history = deque(self.left_motor_history, maxlen=smoothing_amount)
        self.right_motor_history = deque(self.right_motor_history, maxlen=smoothing_amount)
        
        # Apply weighted smoothing
        left_motor = self.smooth_motor_value(left_motor, self.left_motor_history)
        right_motor = self.smooth_motor_value(right_motor, self.right_motor_history)
        
        # Safety bounds
        left_motor = np.clip(left_motor, -1.0, 1.0)
        right_motor = np.clip(right_motor, -1.0, 1.0)
        
        # Publish motor commands
        if not self.shutting_down:
            self.left_motor.publish(left_motor)
            self.right_motor.publish(right_motor)

        # Enhanced display visualization with debug info
        cv2.putText(vis_image, f"Steering: {steering:.2f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(vis_image, f"Speed: {current_speed:.2f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(vis_image, f"Error: {smoothed_error:.2f}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(vis_image, f"Curve: {'Yes' if is_curve else 'No'}", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Show line detection status
        line_status = ""
        if yellow_centroid is not None and white_centroid is not None:
            line_status = "Both lines"
        elif yellow_centroid is not None:
            line_status = "Yellow only"
        elif white_centroid is not None:
            line_status = "White only"
        else:
            line_status = "No lines"
        cv2.putText(vis_image, f"Lines: {line_status}", (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Debug window information
        cv2.putText(debug_image, f"Yellow pixels: {yellow_pixel_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(debug_image, f"White pixels: {white_pixel_count} (right side only)", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(debug_image, f"Total line pixels: {total_line_pixels}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.putText(debug_image, f"ROI: {roi_top}px to {h}px", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(debug_image, f"White exclusion: x < {left_half_boundary}px", (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        if line_spacing is not None:
            cv2.putText(debug_image, f"Line spacing: {line_spacing:.1f}px", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add legend to debug window
        legend_y = h - 120
        cv2.rectangle(debug_image, (10, legend_y), (280, h-10), (0, 0, 0), -1)
        cv2.putText(debug_image, "Legend:", (15, legend_y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(debug_image, "Yellow: Left line", (15, legend_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(debug_image, "White: Right line (right side only)", (15, legend_y + 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(debug_image, "Green: Calculated center", (15, legend_y + 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(debug_image, "Teal/Cyan: Estimated boundary", (15, legend_y + 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 0), 1)
        cv2.putText(debug_image, "Magenta: White line exclusion zone", (15, legend_y + 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        
        cv2.imshow(self._window, vis_image)
        cv2.imshow(self._debug_window, debug_image)
        cv2.waitKey(1)

    def find_line_centroid(self, mask):
        """Find the centroid of the largest contour in the mask"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate moments and centroid
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return None
            
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return (cX, cY)

if __name__ == '__main__':
    node = CameraReaderNode(node_name='camera_reader_node')
    rospy.spin()