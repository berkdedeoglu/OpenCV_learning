import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("yol3.jpg")

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_mask(img):
    height, width = image_gray.shape[:2]
    mask = np.ones((height,width), dtype = np.uint8) * 255
    half_height = height // 2
    half_half_height = half_height // 2
    mask[:half_half_height] = 0
    masked_image = cv2.bitwise_and(img, image_gray, mask=mask)
    return masked_image

masked_image = apply_mask(image_gray)

threshold_value = 200

thres, black_and_white_image = cv2.threshold(masked_image, threshold_value, 255, cv2.THRESH_BINARY)

def detect_and_draw_lines(img):
    edges = cv2.Canny(black_and_white_image, 50, 150, apertureSize=3)

    white_pixels = np.argwhere(black_and_white_image == 255)

    lines = []
    for pixel in white_pixels:
        x, y = pixel[1], pixel[0]
        lines.append([[x, y, x, y]])  

    lines = np.array(lines)
    lines = cv2.HoughLinesP(black_and_white_image, 1, np.pi/180, threshold=20)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1,y1), (x2,y2), (0, 0, 255), 5)
    return image

linesfinded_image = detect_and_draw_lines(black_and_white_image) 

cv2.imshow("Masked Image", masked_image)
cv2.imshow("Black And White", black_and_white_image)
cv2.imshow("Lines Tracked", linesfinded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

