import numpy as np
import cv2
import sys

if __name__ == "__main__":
    cap = cv2.VideoCapture(sys.argv[1])

    frame_num = 0
    while cap.isOpened():
        _, frame = cap.read()

        cv2.imwrite("frame/frame{}.png".format(frame_num), frame)
        frame_num = frame_num + 1

    cap.release()
