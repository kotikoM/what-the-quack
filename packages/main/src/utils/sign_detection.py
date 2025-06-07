#!/usr/bin/env python3
import cv2
import numpy as np
from collections import deque, Counter

class SignDetector:
    road_signs = {
        "68242774271": "Stop Sign",
        "68443584895": "Turn Right",
        "68717272703": "Turn Left"
    }

    def __init__(self):
        self.id_history = deque(maxlen=10)
        self.stable_id = None

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
        dst = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, M, (width, height))

    @staticmethod
    def extract_6x6_id(warped):
        warped = cv2.resize(warped, (120, 120))
        _, binary = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        id_bits = ""
        cell_size = 20
        for row in range(6):
            for col in range(6):
                cell = binary[row*cell_size:(row+1)*cell_size, col*cell_size:(col+1)*cell_size]
                mean = np.mean(cell)
                id_bits += "1" if mean < 128 else "0"
        return int(id_bits, 2), id_bits

    def process_image(self, frame):
        height, width = frame.shape[:2]
        right_half = frame[:, width // 2:]
        gray = cv2.cvtColor(right_half, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 4)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected = False
        raw_id_display = None
        x_offset = width // 2

        for cnt in contours:
            # Draw all contours for debugging
            cv2.drawContours(frame[:, x_offset:], [cnt], -1, (255, 200, 100), 1)

            approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)

            if len(approx) == 4 and cv2.isContourConvex(approx):
                area = cv2.contourArea(approx)
                if not (1000 < area < 100000):  # Filter out too small/large
                    continue

                pts = approx.reshape(4, 2)
                w = np.linalg.norm(pts[0] - pts[1])
                h = np.linalg.norm(pts[1] - pts[2])
                aspect_ratio = w / h if h != 0 else 0
                if not 0.8 < aspect_ratio < 1.2:
                    continue  # Only accept near-square shapes

                try:
                    warped = self.four_point_transform(gray, pts)

                    # Basic perspective validation: brightness/edge check
                    if np.mean(warped) < 40 or np.mean(warped) > 220:
                        continue

                    tag_id, tag_bits = self.extract_6x6_id(warped)
                    self.id_history.append(tag_id)

                    most_common_id, count = Counter(self.id_history).most_common(1)[0]
                    if count > 6:
                        self.stable_id = most_common_id

                    if self.stable_id is not None:
                        raw_id_display = str(self.stable_id)
                        detected = True

                    cx, cy = np.mean(pts, axis=0).astype(int)
                    pts += np.array([x_offset, 0])
                    cx += x_offset

                    cv2.drawContours(frame, [pts.astype(int)], -1, (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {raw_id_display}", (cx - 40, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                except Exception:
                    continue

        if not detected:
            cv2.putText(frame, "Detecting...", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return {
            "stable_id": self.stable_id,
            "sign_name": raw_id_display,
            "frame": frame
        }
