import cv2 as cv
import numpy as np
import sys

img = cv.imread('red_ball.jpg')
if img is None:
    print('no image')
    sys.exit(1)

cv.imshow('raw image', img)

#RGB
#col = (255, 0, 0)

#cvtColor(src, mode)
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow('HSV conversion', hsv)

mask0_low = np.array([0, 150, 0], np.uint8)
mask0_high = np.array([10, 255, 255], np.uint8)
mask0 = cv.inRange(hsv, mask0_low, mask0_high)
cv.imshow('mask0', mask0)

mask1_low = np.array([165, 150, 0], np.uint8)
mask1_high = np.array([179, 255, 255], np.uint8)
mask1 = cv.inRange(hsv, mask1_low, mask1_high)
cv.imshow('mask1', mask1)

combined_mask = mask0 | mask1
cv.imshow('combined_mask', combined_mask)

kernel = np.ones((5, 5), np.uint8)
clean_mask = cv.morphologyEx(combined_mask, cv.MORPH_OPEN, kernel)
clean_mask = cv.morphologyEx(clean_mask, cv.MORPH_CLOSE, kernel)
cv.imshow('clean_mask', clean_mask)

contours, _ = cv.findContours(clean_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

largest_contour = max(contours, key=cv.contourArea)

cv.drawContours(img, [largest_contour], -1, (255, 255, 255), 2)

M = cv.moments(largest_contour)
if M['m00'] != 0:
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    cv.circle(img, (cx, cy), 4, (255, 255, 255), -1)

    cv.putText(img, 'Red ball', (cx - 40, cy - 10),
            cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)

    print('centre:', cx, cy)

cv.imshow('result', img)
cv.waitKey(0)
cv.destroyAllWindows()