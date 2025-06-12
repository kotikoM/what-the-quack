#!/usr/bin/env python3
import cv2
import apriltag
import numpy as np

class SignDetector:
    def detect_sign(self, frame):
        height, width = frame.shape[:2]

        right_half = frame[:, width // 2:]
        gray = cv2.cvtColor(right_half, cv2.COLOR_BGR2GRAY)

        options = apriltag.DetectorOptions(families="tag36h11")
        detector = apriltag.Detector(options)
        detections = detector.detect(gray)
        visual_frame = right_half.copy()

        filtered_detections = []
        for detection in detections:
            corners = np.array(detection.corners, dtype=np.float32)
            area = cv2.contourArea(corners)
            if area < 4000:
                continue

            filtered_detections.append(detection)

            for i in range(4):
                pt1 = tuple(map(int, detection.corners[i]))
                pt2 = tuple(map(int, detection.corners[(i + 1) % 4]))
                cv2.line(visual_frame, pt1, pt2, (0, 255, 0), 2)

            center = tuple(map(int, detection.center))
            cv2.circle(visual_frame, center, 5, (0, 0, 255), -1)
            cv2.putText(visual_frame, str(detection.tag_id), center,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        return {
            "frame": visual_frame,
            "detections": filtered_detections
        }
