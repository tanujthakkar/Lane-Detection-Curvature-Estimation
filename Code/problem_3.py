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
        hls_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        h, l, s = cv2.split(hls_frame)

        _, l_thresh = cv2.threshold(h, 120, 255, cv2.THRESH_BINARY)
        thresh_blur = cv2.GaussianBlur(l_thresh, (5, 5), 0)

        sobel_x = np.absolute(cv2.Sobel(thresh_blur, cv2.CV_32F, 1, 0, 3))
        sobel_y = np.absolute(cv2.Sobel(thresh_blur, cv2.CV_32F, 0, 1, 3))
        mag = np.sqrt(sobel_x**2 + sobel_y**2)
        binary = np.ones_like(mag)
        binary[(mag >= 110) & (mag <= 255)] = 1

        _, s_thresh = cv2.threshold(s, 100, 255, cv2.THRESH_BINARY)
        _, r_thresh = cv2.threshold(frame[:,:,2], 120, 255, cv2.THRESH_BINARY)

        rs_binary = cv2.bitwise_and(s_thresh, r_thresh)
        filtered_lines = cv2.bitwise_or(rs_binary, np.uint8(binary))

        return filtered_lines

    def __mask_roi(self, frame: np.array, corners: np.array) -> np.array:
        mask = np.zeros_like(frame)
        cv2.fillPoly(mask, [corners], 255)
        roi = cv2.bitwise_and(frame, mask)

        return roi

    def __birds_eye_view(self, frame: np.array, corners: np.array, height: int, width: int) -> np.array:

        padding = width * .25
        corners_bev = np.array([[0, 0],
                            [0, width],
                            [height, width],
                            [height, 0]]) # Corners of the warped birds-eye-view
        H = cv2.getPerspectiveTransform(np.float32(corners), np.float32(corners_bev))
        H_inv = np.linalg.inv(H)

        birds_eye_view = cv2.warpPerspective(frame, H, (750, frame.shape[0]))

        return birds_eye_view

    def __estimate_curvature(self, frame: np.array) -> np.array:

        # frame_blur = cv2.GaussianBlur(frame, (5, 5), 0)
        frame_filtered = self.__filter_lines(frame)

        corners = np.array([[590, 450],
                            [220, 680],
                            [1130, 680],
                            [735, 450]]) # Corners of the region of interest
        cv2.polylines(frame, [corners], True, (0,0,255), 2)
        frame_roi = self.__mask_roi(frame_filtered, corners)

        birds_eye_view = self.__birds_eye_view(frame_roi, corners, frame_roi.shape[0], frame_roi.shape[1])

        cv2.imshow("Frame", frame)
        cv2.imshow("Lane", frame_roi)
        cv2.imshow("Birds Eye View", birds_eye_view)
        cv2.waitKey()


    def process_video(self, visualize: bool = False) -> None:
        video = cv2.VideoCapture(self.video_path)
        ret = True

        while(ret):
            ret, frame = video.read()
            self.__estimate_curvature(frame)

            if(visualize):
                cv2.imshow("Result", result)
                cv2.imshow("ROI", frame_roi)
                cv2.waitKey(0)

def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--VideoPath', type=str, default="../Data/challenge.mp4", help='Path to input video')
    Parser.add_argument('--Visualize', action='store_true', help='Toggle visualization')
    
    Args = Parser.parse_args()
    video_path = Args.VideoPath
    visualize = Args.Visualize
    
    CE = CurvatureEstimator(video_path)
    CE.process_video(visualize)

if __name__ == '__main__':
    main()