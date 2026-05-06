import cv2 as cv
import numpy as np
import sys

cap = cv.VideoCapture('rgb_ball_720.mp4')

if not cap.isOpened():
    print('no video')
    sys.exit(1)

last_center = None
last_radius = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    mask0_low = np.array([0, 170, 120], np.uint8)
    mask0_high = np.array([8, 255, 255], np.uint8)
    mask0 = cv.inRange(hsv, mask0_low, mask0_high)

    mask1_low = np.array([172, 170, 120], np.uint8)
    mask1_high = np.array([179, 255, 255], np.uint8)
    mask1 = cv.inRange(hsv, mask1_low, mask1_high)

    combined_mask = mask0 | mask1

    kernel = np.ones((5, 5), np.uint8)
    clean_mask = cv.morphologyEx(combined_mask, cv.MORPH_OPEN, kernel)
    clean_mask = cv.morphologyEx(clean_mask, cv.MORPH_CLOSE, kernel)

    contours, _ = cv.findContours(clean_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    found = False

    if len(contours) > 0:
        largest_contour = max(contours, key=cv.contourArea)

        area = cv.contourArea(largest_contour)

        if area > 250:
            M = cv.moments(largest_contour)

            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                (x, y), radius = cv.minEnclosingCircle(largest_contour)

                cv.drawContours(frame, [largest_contour], -1, (255, 255, 255), 2)
                cv.circle(frame, (cx, cy), 4, (255, 255, 255), -1)
                cv.putText(frame, 'Red ball', (cx - 40, cy - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)

                last_center = (cx, cy)
                last_radius = int(radius)
                found = True

    if not found and last_center is not None:
        cv.circle(frame, last_center, 6, (255, 255, 255), -1)
        cv.line(frame,
                (last_center[0] - 15, last_center[1] - 15),
                (last_center[0] + 15, last_center[1] + 15),
                (255, 255, 255), 2)
        cv.line(frame,
                (last_center[0] - 15, last_center[1] + 15),
                (last_center[0] + 15, last_center[1] - 15),
                (255, 255, 255), 2)
        cv.putText(frame, 'Red ball not in view',
                   (last_center[0] - 80, last_center[1] - 20),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)

    cv.imshow('video tracking', frame)

    key = cv.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv.destroyAllWindows()