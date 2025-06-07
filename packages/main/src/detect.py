#!/usr/bin/env python3
import cv2
import numpy as np
import time
import rospy
from collections import deque, Counter
from std_msgs.msg import String

road_signs = {
    "68242774271": 1,
    "68443584895": 3,
    "68717272703": 3
}

# --- Helper Functions ---
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (width, height))

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

def main():
    rospy.init_node('sign_detector')
    pub = rospy.Publisher('/detected_sign', String, queue_size=10)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        rospy.logerr("Could not open camera.")
        return

    id_history = deque(maxlen=10)
    stable_id = None
    last_printed_id = None
    last_print_time = 0

    rospy.loginfo("Sign detector running. Press 'q' to exit window.")

    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if not ret:
            break

        height, width = frame.shape[:2]
        right_half = frame[:, width // 2:]
        gray = cv2.cvtColor(right_half, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.05 * cv2.arcLength(cnt, True), True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                area = cv2.contourArea(approx)
                if area > 1000:
                    pts = approx.reshape(4, 2)
                    try:
                        warped = four_point_transform(gray, pts)
                        tag_id, tag_bits = extract_6x6_id(warped)

                        if str(tag_id) not in road_signs:
                            rospy.logwarn(f"Unknown tag detected: {tag_id} (bits: {tag_bits})")

                        cx, cy = np.mean(pts, axis=0).astype(int)
                        pts += np.array([width // 2, 0])
                        cx += width // 2
                        cv2.drawContours(frame, [pts.astype(int)], -1, (0, 255, 0), 2)
                        id_history.append(tag_id)

                        most_common_id, count = Counter(id_history).most_common(1)[0]
                        if count > 6:
                            stable_id = most_common_id

                        if stable_id is not None:
                            sign_name = road_signs.get(str(stable_id), "Not Detected")
                            cv2.putText(frame, f"ID: {sign_name}", (cx-40, cy),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    except:
                        continue

        current_time = time.time()
        if stable_id is not None and (stable_id != last_printed_id or current_time - last_print_time >= 2):
            sign_name = road_signs.get(str(stable_id), "Not Detected")
            pub.publish(sign_name)
            rospy.loginfo(f"Published sign: {sign_name}")
            last_printed_id = stable_id
            last_print_time = current_time

        if stable_id is None:
            cv2.putText(frame, "Detecting...", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Tag Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()