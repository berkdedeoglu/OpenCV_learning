import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture("yolvideo.mp4")

def apply_mask(frame):
    height, width = frame.shape[:2]
    mask = np.ones((height, width), dtype=np.uint8) * 255
    half_height = height // 2
    half_half_height = half_height // 2
    mask[:half_half_height] = 0
    masked_image = cv2.bitwise_and(frame, frame, mask=mask)
    return masked_image

threshold_value = 200

def detect_and_draw_lines(frame, original_frame):
    edges = cv2.Canny(frame, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=20)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(original_frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
    return original_frame

if not cap.isOpened():
    print("Error opening video file")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if ret:

        avarageValue = np.float32(frame)
        lines_frame_slowed = cv2.accumulateWeighted(frame, avarageValue, 0.5)
        resultingFrame = cv2.convertScaleAbs(avarageValue)

        gray_frame = cv2.cvtColor(resultingFrame, cv2.COLOR_BGR2GRAY)
        masked_frame = apply_mask(gray_frame)
        thres, black_and_white_frame = cv2.threshold(masked_frame, threshold_value, 255, cv2.THRESH_BINARY)

        lines_frame = detect_and_draw_lines(black_and_white_frame, frame.copy())
        lines_frame_resized = cv2.resize(lines_frame, (580, 350))

        cv2.putText(lines_frame_resized,"Serit Takip", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)

        cv2.imshow('Detected Lines', lines_frame_resized)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
