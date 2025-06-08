#!/usr/bin/env python3
import cv2
import rospy
import numpy as np
from collections import deque


class SignDetector:
    road_signs = {
        "101100_010110_111001_111101_111110_110000": "Stop",
        "101000_000010_001000_111001_100111_100010": "Parking",
        "011011_100000_011110_011100_110010_011011": "Slow Down"
    }

    def __init__(self):
        self.id_history = deque(maxlen=10)
        self.stable_id = None

    def hamming_distance(s1, s2):
        return sum(c1 != c2 for c1, c2 in zip(s1, s2))

    def get_closest_sign(self, tag_id, max_distance=5):
        closest_id = None
        min_distance = float('inf')
        for known_id in self.road_signs:
            distance = self.hamming_distance(tag_id, known_id)
            if distance < min_distance and distance <= max_distance:
                min_distance = distance
                closest_id = known_id
        return closest_id

    @staticmethod
    def extract_6x6_id(warped):
        # TODO: Enhance white black square detection,
        warped = cv2.resize(warped, (140, 140))
        _, binary = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        id_bits = ""
        cell_size = 20
        for row in range(1, 7):
            for col in range(1, 7):
                cell = binary[row * cell_size:(row + 1) * cell_size,
                       col * cell_size:(col + 1) * cell_size]
                mean = np.mean(cell)
                id_bits += "1" if mean > 128 else "0"

        return id_bits, binary

    @staticmethod
    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    @staticmethod
    def four_point_transform(image, pts):
        rect = SignDetector.order_points(pts)
        (tl, tr, br, bl) = rect
        width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
        height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
        dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, M, (width, height))

    @staticmethod
    def has_internal_black_border(warped, border_width_ratio=0.2, black_threshold=50, coverage=0.8):
        h, w = warped.shape
        bw = int(border_width_ratio * min(h, w))
        outer = np.zeros_like(warped, dtype=np.uint8)
        cv2.rectangle(outer, (0, 0), (w - 1, h - 1), 255, thickness=bw)
        border_pixels = warped[outer == 255]
        num_black = np.sum(border_pixels < black_threshold)
        total = len(border_pixels)

        return num_black / total >= coverage if total > 0 else False

    def process_image(self, frame):
        height, width = frame.shape[:2]

        # Work on two copies
        contours_frame = frame.copy()
        squares_frame = frame.copy()

        # Focus on right half
        right_half = frame[:, width // 2:]
        gray = cv2.cvtColor(right_half, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 4)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        warped_images = []
        for contour in contours:
            # Draw all contours in blue on right half of contours_frame
            cv2.drawContours(contours_frame[:, width // 2:], [contour], -1, (255, 255, 255), 1)

            area = cv2.contourArea(contour)
            if area < 1000 or area > 100000:
                continue

            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) == 4:
                pts = approx.reshape(4, 2)
                side_lengths = [
                    np.linalg.norm(pts[i] - pts[(i + 1) % 4])
                    for i in range(4)
                ]
                min_side = min(side_lengths)
                max_side = max(side_lengths)
                aspect_ratio = max_side / min_side if min_side > 0 else float("inf")

                if aspect_ratio > 2.0:
                    continue

                cv2.drawContours(squares_frame[:, width // 2:], [approx], -1, (0, 255, 0), 2)
                for point in approx:
                    x, y = point[0]
                    cv2.circle(squares_frame[:, width // 2:], (x, y), 4, (0, 255, 0), -1)

                warped = SignDetector.four_point_transform(gray, pts)
                if SignDetector.has_internal_black_border(warped):
                    warped_images.append(warped)


        for w in warped_images:
            id_bits = SignDetector.extract_6x6_id(w)
            rospy.loginfo(id_bits)

        return {
            "contours_frame": contours_frame,
            "squares_frame": squares_frame,
            "warped_images": warped_images
        }
