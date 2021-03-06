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

    def __init__(self, video_path: str, save_path: str) -> None:
        self.video_path = video_path
        self.save_path = save_path
        if(not os.path.exists(self.save_path)):
            os.makedirs(self.save_path, exist_ok=True)

    def __filter_lines(self, frame: np.array) -> np.array:
        hls_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        h, l, s = cv2.split(hls_frame)

        _, s_thresh = cv2.threshold(s, 100, 255, cv2.THRESH_BINARY)
        _, r_thresh = cv2.threshold(frame[:,:,2], 120, 255, cv2.THRESH_BINARY)

        filtered_lines = cv2.bitwise_and(s_thresh, r_thresh)

        return filtered_lines

    def __mask_roi(self, frame: np.array, corners: np.array) -> np.array:
        mask = np.zeros_like(frame)
        cv2.fillPoly(mask, [corners], 255)
        roi = cv2.bitwise_and(frame, mask)

        return roi

    def __birds_eye_view(self, frame: np.array, corners: np.array, height: int, width: int) -> np.array:

        corners_bev = np.array([[320, 1],
                                [320, 980],
                                [920, 980],
                                [920, 1]]) # Corners of the warped birds-eye-view
        H = cv2.getPerspectiveTransform(np.float32(corners), np.float32(corners_bev))
        birds_eye_view = cv2.warpPerspective(frame, H, (frame.shape[1], 1000))

        return birds_eye_view, H

    def __calculate_histogram_peaks(self, frame: np.array, visualize: bool = False) -> np.array:

        location = np.arange(frame.shape[1], dtype=int)
        histogram = np.sum(frame[int(frame.shape[0]/2):,:], axis=0)

        midpoint = histogram.shape[0]//2
        left_peak = np.argmax(histogram[:midpoint])
        right_peak = np.argmax(histogram[midpoint:]) + midpoint

        if(visualize):
            fig = plt.figure()
            plt.xlabel("Pixel Location")
            plt.ylabel("Pixel Count")
            plt.plot(location, histogram)
            fig.set_size_inches(5, 4)
            plt.savefig("hist.png", dpi=100, bbox_inches='tight')
            plt.show()

        return histogram, left_peak, right_peak

    def __sliding_window(self, frame: np.array, histogram_peaks: list, num_windows: int) -> np.array:

        frame_sliding_window = convert_three_channel(np.copy(frame))
        # print(frame_sliding_window.shape)

        window_height = frame.shape[0]//num_windows
        window_width = frame.shape[1]//12
        # print("Window H, W:", window_height, window_width)

        left_peak, right_peak = histogram_peaks
        left_window_mean = left_peak
        right_window_mean = right_peak

        left_centroids = list()
        right_centroids = list()

        height = frame.shape[0]-1 - window_height
        for window in range(num_windows-1):
            left_window = frame[height:height + window_height, left_window_mean - (window_width//2):left_window_mean + (window_width//2)]
            if(len(np.where(left_window == 255)[1]) > 10):
                left_window_mean = int(np.mean(np.where(left_window == 255)[1])) + left_window_mean - (window_width//2)
            left_centroids.append([height + window_height//2, left_window_mean])
            cv2.circle(frame_sliding_window, (left_window_mean, height + window_height//2), 10, (255,0,0), 5)
            # cv2.rectangle(frame_sliding_window, (left_window_mean - (window_width//2), height), (left_window_mean + (window_width//2), height + window_height), (0,255,0), 2)

            right_window = frame[height:height + window_height, right_window_mean - (window_width//2):right_window_mean + (window_width//2)]
            if(len(np.where(right_window == 255)[1]) > 10):
                right_window_mean = int(np.mean(np.where(right_window == 255)[1])) + right_window_mean - (window_width//2)
            right_centroids.append([height + window_height//2, right_window_mean])
            cv2.circle(frame_sliding_window, (right_window_mean, height + window_height//2), 10, (0,255,0), 5)
            # cv2.rectangle(frame_sliding_window, (right_window_mean - (window_width//2), height), (right_window_mean + (window_width//2), height + window_height), (0,255,0), 2)

            height -= window_height

        centroids = np.array([left_centroids,
                              right_centroids])

        return centroids, frame_sliding_window

    def __fit_lines(self, frame: np.array, centroids: np.array, sliding_average: int = 10) -> np.array:

        frame_lane_lines = np.copy(frame)

        left_fit = np.polyfit(centroids[0,:,0], centroids[0,:,1], 2)
        right_fit = np.polyfit(centroids[1,:,0], centroids[1,:,1], 2)

        if(len(self.line_fits[0]) >= sliding_average):
            self.line_fits[0].pop(0)
            self.line_fits[1].pop(0)
        self.line_fits[0].append(left_fit)
        self.line_fits[1].append(right_fit)

        left_fit = np.mean(self.line_fits[0], axis=0)
        right_fit = np.mean(self.line_fits[1], axis=0)

        x = np.linspace(0, frame.shape[0]-1, frame.shape[0])
        y = left_fit[0]*(x**2) + left_fit[1]*x + left_fit[2]
        left_lane_line = np.array([y, x], dtype=int).transpose()

        y = right_fit[0]*(x**2) + right_fit[1]*x + right_fit[2]
        right_lane_line = np.array([y, x], dtype=int).transpose()

        cv2.polylines(frame_lane_lines, [left_lane_line], False, (0, 0, 255), 4)
        cv2.polylines(frame_lane_lines, [right_lane_line], False, (0, 255, 255), 4)

        return left_fit, left_lane_line, right_fit, right_lane_line, frame_lane_lines

    def __project_lane(self, frame: np.array, birds_eye_view: np.array, lane_lines: np.array, homography: np.array) -> np.array:

        frame_projected = convert_three_channel(np.zeros_like(birds_eye_view))
        cv2.fillPoly(frame_projected, [lane_lines], (0, 255, 0))

        H_inv = np.linalg.inv(homography)

        frame_projected = cv2.warpPerspective(frame_projected, H_inv, (frame.shape[1], frame.shape[0]))
        frame_projected = cv2.addWeighted(frame, 1, frame_projected, 0.4, 0)

        return frame_projected

    def __estimate_curvature(self, frame: np.array) -> np.array:

        # frame = np.array(np.flip(frame, axis=1))
        frame_filtered = self.__filter_lines(frame)

        h, w, _ = frame.shape
        corners = np.array([[570, 470],
                            [220, 680],
                            [1200, 680],
                            [750, 470]]) # Corners of the region of interest

        # cv2.polylines(frame, [corners], True, (0,0,255), 2)
        # frame_roi = self.__mask_roi(frame_filtered, corners)

        birds_eye_view, H = self.__birds_eye_view(frame_filtered, corners, frame_filtered.shape[0], frame_filtered.shape[1])
        histogram, left_peak, right_peak = self.__calculate_histogram_peaks(birds_eye_view)
        centroids, frame_sliding_window = self.__sliding_window(birds_eye_view, [left_peak, right_peak], 20)
        left_fit, left_lane_line, right_fit, right_lane_line, frame_lane_lines = self.__fit_lines(frame_sliding_window, centroids)

        left_curvature = ((1+(2*left_fit[0]*frame_sliding_window.shape[0]*(30/frame_sliding_window.shape[0]) + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
        right_curvature = ((1+(2*right_fit[0]*frame_sliding_window.shape[0]*(30/frame_sliding_window.shape[0]) + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
        curvature_diff = left_curvature - right_curvature

        frame_projected = self.__project_lane(frame, birds_eye_view, np.vstack((left_lane_line, np.flipud(right_lane_line))), H)
        
        cv2.putText(frame_projected, "Left Curvature: {:.2f}".format(left_curvature), (20,30), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame_projected, "Right Curvature: {:.2f}".format(right_curvature), (920,30), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame_projected, "Avg. Curvature: {:.2f}".format((left_curvature+right_curvature)/2), (w//2-125,30), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 255), 2, cv2.LINE_AA)

        if(curvature_diff > 0):
            cv2.putText(frame_projected, 'Turn Right', (w//2-50,60), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 255), 2, cv2.LINE_AA)
        elif(curvature_diff < 0):
            cv2.putText(frame_projected, 'Turn Left', (w//2-50,60), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame_projected, 'Go Straight', (w//2,60), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, 'Frame', (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), 2, cv2.LINE_AA)
        frame = cv2.resize(frame, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC)

        frame_filtered = cv2.putText(convert_three_channel(frame_filtered), 'Lane Filtering', (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), 2, cv2.LINE_AA)
        frame_filtered = cv2.resize(frame_filtered, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC)
        
        birds_eye_view = cv2.putText(convert_three_channel(birds_eye_view), 'Birds Eye View', (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), 2, cv2.LINE_AA)
        birds_eye_view = cv2.resize(birds_eye_view, None, fx=0.4, fy=0.432, interpolation=cv2.INTER_CUBIC)
        
        cv2.putText(frame_lane_lines, 'Curvature Estimation', (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), 2, cv2.LINE_AA)
        frame_lane_lines = cv2.resize(frame_lane_lines, None, fx=0.4, fy=0.432, interpolation=cv2.INTER_CUBIC)
        # frame_projected = cv2.resize(frame_projected, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_CUBIC)

        top_row = np.hstack((frame, frame_filtered))
        bottom_row = np.hstack((birds_eye_view, frame_lane_lines))
        pipeline = np.vstack((top_row, bottom_row))
        result = np.hstack((frame_projected, pipeline))
        result = cv2.resize(result, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_CUBIC)

        # cv2.imshow("Frame", frame_projected)
        # cv2.imshow("Lane", temp)
        # cv2.imshow("Birds Eye View", birds_eye_view)
        # cv2.imshow("Sliding Window", frame_sliding_window)
        # cv2.imshow("Lane Lines", frame_lane_lines)
        # cv2.waitKey()

        return result

    def process_video(self, save_output: bool = False, visualize: bool = False) -> None:
        video = cv2.VideoCapture(self.video_path)
        ret = True

        self.line_fits = [[],[]] # Sliding mean of polynomial coefficients

        if(save_output):
            ret, frame = video.read()
            result = self.__estimate_curvature(frame)
            
            save_file = os.path.join(self.save_path, self.video_path.split('/')[-1].split('.')[0]) + '_processed.mp4'
            video_writer = cv2.VideoWriter(save_file, cv2.VideoWriter_fourcc('M','J','P','G'), 24, (result.shape[1], result.shape[0]))
            video_writer.write(result)

        while(ret):
            try:
                ret, frame = video.read()
                result = self.__estimate_curvature(frame)

                if(save_output):
                    video_writer.write(result)

                if(visualize):
                    cv2.imshow("Result", result)
                    cv2.waitKey(3)
            except Exception as e:
                # print(e)
                pass

        if(save_output):
            cv2.destroyAllWindows()
            video_writer.release()

def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--VideoPath', type=str, default="../Data/challenge.mp4", help='Path to input video')
    Parser.add_argument('--Save', action='store_true', help='Toggle to save output')
    Parser.add_argument('--SavePath', type=str, default="../Result/", help='Path to results folder')
    Parser.add_argument('--Visualize', action='store_true', help='Toggle visualization')
    
    Args = Parser.parse_args()
    video_path = Args.VideoPath
    save_output = Args.Save
    save_path = Args.SavePath
    visualize = Args.Visualize
    
    CE = CurvatureEstimator(video_path, save_path)
    CE.process_video(save_output, visualize)

if __name__ == '__main__':
    main()