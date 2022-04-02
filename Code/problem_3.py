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
import time

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

    def __calculate_histogram_peaks(self, frame: np.array, visualize: bool = False) -> np.array:

        location = np.arange(frame.shape[1], dtype=int)
        histogram = np.sum(frame[int(frame.shape[0]/2):,:], axis=0)

        midpoint = histogram.shape[0]//2
        left_peak = np.argmax(histogram[:midpoint])
        right_peak = np.argmax(histogram[midpoint:]) + midpoint

        if(visualize):
            plt.figure()
            plt.title("Histogram")
            plt.xlabel("Pixel Location")
            plt.ylabel("Pixel Count")
            plt.plot(location, histogram)
            plt.show()

        return histogram, left_peak, right_peak

    def __sliding_window(self, frame: np.array, histogram_peaks: list, num_windows: int):

        frame_sliding_window = np.copy(frame)
        # print(frame_sliding_window.shape)

        window_height = frame.shape[0]//num_windows
        window_width = frame.shape[1]//12
        # print("Window H, W:", window_height, window_width)

        left_peak, right_peak = histogram_peaks
        left_window_mean = left_peak
        right_window_mean = right_peak

        height = frame.shape[0]-1 - window_height
        for window in range(num_windows-1):
            left_window = frame[height:height + window_height, left_window_mean - (window_width//2):left_window_mean + (window_width//2)]
            if(len(np.where(left_window == 255)[1]) > 10):
                left_window_mean = int(np.mean(np.where(left_window == 255)[1])) + left_window_mean - (window_width//2)
            cv2.rectangle(frame_sliding_window, (left_window_mean - (window_width//2), height), (left_window_mean + (window_width//2), height + window_height), 255, 2)

            right_window = frame[height:height + window_height, right_window_mean - (window_width//2):right_window_mean + (window_width//2)]
            if(len(np.where(right_window == 255)[1]) > 10):
                right_window_mean = int(np.mean(np.where(right_window == 255)[1])) + right_window_mean - (window_width//2)
            cv2.rectangle(frame_sliding_window, (right_window_mean - (window_width//2), height), (right_window_mean + (window_width//2), height + window_height), 255, 2)

            height -= window_height

        cv2.imshow("Sliding Window", frame_sliding_window)
        cv2.waitKey()

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
        histogram, left_peak, right_peak = self.__calculate_histogram_peaks(birds_eye_view)
        self.__sliding_window(birds_eye_view, [left_peak, right_peak], 20)

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