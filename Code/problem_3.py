#!/usr/env/bin python3

"""
ENPM673 Spring 2022: Perception for Autonomous Robots

Project 2: Problem 3 - Lane Curvature Estimation

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


class CurvatureEstimator:

    def __init__(self, video_path: str) -> None:
        self.video_path = video_path

    def __filter_lines(self, frame: np.array) -> np.array:
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # YELLOW color range values
        yellow_lower = np.array([0, 100, 100])
        yellow_upper = np.array([50, 255, 255])
         
        # WHITE color range values
        white_lower = np.array([10, 0, 170])
        white_upper = np.array([255, 90, 255])
         
        lower_mask = cv2.inRange(hsv_frame, yellow_lower, yellow_upper)
        upper_mask = cv2.inRange(hsv_frame, white_lower, white_upper)
         
        full_mask = lower_mask + upper_mask

        return full_mask

    def __mask_roi(self, frame: np.array, corners: np.array) -> np.array:
        mask = np.zeros_like(frame)
        cv2.fillPoly(mask, [corners], 255)
        roi = cv2.bitwise_and(frame, mask)

        return roi

    def __birds_eye_view(self, frame: np.array, corners: np.array, height: int, width: int) -> np.array:

        corners_bev = np.array([[0, 0],
                            [height, 0],
                            [height, width],
                            [0, width]]) # Corners of the warped birds-eye-view
        H = cv2.getPerspectiveTransform(np.float32(corners), np.float32(corners_bev))
        
        print(H)

    def __estimate_curvature(self, frame: np.array) -> np.array:

        frame_blur = cv2.GaussianBlur(frame, (5, 5), 0)
        frame_filtered = self.__filter_lines(frame_blur)

        corners = np.array([[220, 680],
                            [1130, 680],
                            [735, 450],
                            [590, 450]]) # Corners of the region of interest
        frame_roi = self.__mask_roi(frame_filtered, corners)

        self.__birds_eye_view(frame_roi, corners, 720, 360)

        cv2.imshow("Frame", frame_roi)
        cv2.waitKey()


    def process_video(self) -> None:
        video = cv2.VideoCapture(self.video_path)
        ret = True

        while(ret):
            ret, frame = video.read()
            self.__estimate_curvature(frame)

def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--VideoPath', type=str, default="../Data/challenge.mp4", help='Path to input video')
    Parser.add_argument('--Visualize', action='store_true', help='Toggle visualization')
    
    Args = Parser.parse_args()
    video_path = Args.VideoPath
    
    CE = CurvatureEstimator(video_path)
    CE.process_video()

if __name__ == '__main__':
    main()