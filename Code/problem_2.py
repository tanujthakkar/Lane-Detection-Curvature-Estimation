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

    def __init__(self, video_path: str, save_path: str) -> None:
        self.video_path = video_path
        self.save_path = save_path
        if(not os.path.exists(self.save_path)):
            os.makedirs(self.save_path, exist_ok=True)

    def __mask_roi(self, frame: np.array, corners: np.array) -> np.array:
        mask = np.zeros_like(frame)
        cv2.fillPoly(mask, [corners], 255)
        roi = cv2.bitwise_and(frame, mask)

        return roi

    def __classify_lanes(self, frame: np.array, lanes: list) -> list:

        longest = -np.inf
        gradient = None
        for lane in lanes:
            for x1, y1, x2, y2 in lane:
                dist = np.sqrt((y2-y1)**2 + (x2-x1)**2)
                if(dist >= longest):
                    longest = dist
                    gradient = (y2-y1)/(x2-x1)

        solid_lane = list()
        dashed_lane = list()

        for lane in lanes:
            for x1, y1, x2, y2 in lane:
                if(((y2-y1)/(x2-x1) > 0 and gradient > 0) or ((y2-y1)/(x2-x1) < 0 and gradient < 0)):
                    solid_lane.append([x1, y1])
                    solid_lane.append([x2, y2])
                    # cv2.line(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                else:
                    dashed_lane.append([x1, y1])
                    dashed_lane.append([x2, y2])
                    # cv2.line(frame, (x1, y1), (x2, y2), (0,0,255), 2)

        return [solid_lane, dashed_lane]

    def __fit_lines(self, frame: np.array, lines: list) -> None:

        solid_line_pts = np.array(lines[0])
        dashed_line_pts = np.array(lines[1])

        solid_line_fit = np.polyfit(solid_line_pts[:,1], solid_line_pts[:,0], 1)
        dashed_line_fit = np.polyfit(dashed_line_pts[:,1], dashed_line_pts[:,0], 1)

        x_min, y_min = np.min(solid_line_pts, axis=0)
        x_max, y_max = np.max(solid_line_pts, axis=0)
        x_min = int(solid_line_fit[0]*y_min + solid_line_fit[1])
        x_max = int(solid_line_fit[0]*y_max + solid_line_fit[1])
        cv2.line(frame, (x_min, y_min), (x_max, y_max), (0,225,0), 2)

        x_min, _ = np.min(dashed_line_pts, axis=0)
        x_max, _ = np.max(dashed_line_pts, axis=0)
        x_min = int(dashed_line_fit[0]*y_min + dashed_line_fit[1])
        x_max = int(dashed_line_fit[0]*y_max + dashed_line_fit[1])
        cv2.line(frame, (x_min, y_min), (x_max, y_max), (0,0,225), 2)


    def __detect_lines(self, frame: np.array) -> np.array:

        # frame = np.flip(frame, axis=1)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_blur = cv2.GaussianBlur(frame_gray, (5, 5), 0)
        frame_canny = cv2.Canny(frame_blur, 75, 150)

        corners = np.array([[125, frame_gray.shape[0]],
                            [895, frame_gray.shape[0]],
                            [520, 320],
                            [440, 320]]) # Corners of the region of interest
        frame_roi = self.__mask_roi(frame_canny, corners)

        lanes = cv2.HoughLinesP(frame_roi, 2, np.pi/180, 20, np.array([]), minLineLength=25, maxLineGap=10)
        frame_lanes = np.copy(frame)
        cv2.putText(frame_lanes, 'Solid', (20,20), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame_lanes, 'Dashed', (20,40), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 2, cv2.LINE_AA)

        lanes = self.__classify_lanes(frame_lanes, lanes)
        self.__fit_lines(frame_lanes, lanes)

        return frame_lanes, frame_roi

    def process_frame(self, frame: np.array) -> np.array:
        result, frame_roi = self.__detect_lines(frame)

        return result, frame_roi

    def process_video(self, save_output: bool = False, visualize: bool = False) -> None:
        video = cv2.VideoCapture(self.video_path)
        ret = True

        if(save_output):
            ret, frame = video.read()
            result, frame_roi = self.__detect_lines(frame)
            
            save_file = os.path.join(self.save_path, self.video_path.split('/')[-1].split('.')[0]) + '_processed.mp4'
            video_writer = cv2.VideoWriter(save_file, cv2.VideoWriter_fourcc('M','J','P','G'), 24, (result.shape[1], result.shape[0]))
            video_writer.write(result)

        while(ret):
            try:
                ret, frame = video.read()
                result, frame_roi = self.__detect_lines(frame)

                if(save_output):
                    video_writer.write(result)

                if(visualize):
                    # cv2.imshow("Frame", frame)
                    cv2.imshow("Result", result)
                    cv2.waitKey(3)
            except Exception as e:
                # print("Exception: ", e)
                pass

        if(save_output):
            cv2.destroyAllWindows()
            video_writer.release()


def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--VideoPath', type=str, default="../Data/whiteline.mp4", help='Path to input video')
    Parser.add_argument('--Save', action='store_true', help='Toggle to save output')
    Parser.add_argument('--SavePath', type=str, default="../Result/", help='Path to results folder')
    Parser.add_argument('--Visualize', action='store_true', help='Toggle visualization')
    
    Args = Parser.parse_args()
    video_path = Args.VideoPath
    save_output = Args.Save
    save_path = Args.SavePath
    visualize = Args.Visualize

    LD = LaneDetector(video_path, save_path)
    LD.process_video(save_output, visualize)

if __name__ == '__main__':
    main()