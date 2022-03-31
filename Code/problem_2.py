#!/usr/env/bin python3

"""
ENPM673 Spring 2022: Perception for Autonomous Robots

Project 2: Problem 2 - Lane Detection

Author(s):
Tanuj Thakkar (tanuj@umd.edu)
M. Engg Robotics
University of Maryland, College Park
"""

# Importing modules
import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse

from utils import *



class LaneDetector:

    def __init__(self, video_path: str) -> None:
        self.video_path = video_path

    def __mask_roi(self, frame: np.array, corners: np.array) -> np.array:
        mask = np.zeros_like(frame)
        cv2.fillPoly(mask, [corners], 255)
        roi = cv2.bitwise_and(frame, mask)

        return roi

    def __detect_lines(self, frame: np.array) -> np.array:

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_blur = cv2.GaussianBlur(frame_gray, (5, 5), 0)
        frame_canny = cv2.Canny(frame_blur, 50, 150)

        corners = np.array([[125, frame_gray.shape[0]],
                            [895, frame_gray.shape[0]],
                            [520, 320],
                            [440, 320]])
        frame_roi = self.__mask_roi(frame_canny, corners)

        cv2.imshow("Frame - Canny", frame_roi)
        cv2.waitKey()


    def process_video(self) -> None:
        video = cv2.VideoCapture(self.video_path)
        ret = True

        while(ret):
            ret, frame = video.read()

            self.__detect_lines(frame)


def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--VideoPath', type=str, default="../Data/whiteline.mp4", help='Path to input video')
    Parser.add_argument('--Visualize', action='store_true', help='Toggle visualization')
    
    Args = Parser.parse_args()
    video_path = Args.VideoPath
    
    LD = LaneDetector(video_path)
    LD.process_video()

if __name__ == '__main__':
    main()