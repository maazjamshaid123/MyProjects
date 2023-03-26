import cv2
import numpy as np

img = cv2.imread('GreenScreen.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_green = np.array([35, 50, 50])
upper_green = np.array([90, 255, 255])
mask = cv2.inRange(hsv, lower_green, upper_green)
result = cv2.bitwise_and(img, img, mask=mask)
cv2.imwrite('result.jpg', result)
