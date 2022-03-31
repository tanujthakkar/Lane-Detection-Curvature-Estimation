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


    def process_video(self) -> None:
        video = cv2.VideoCapture(self.video_path)
        ret = True

        while(ret):
            ret, frame = video.read()

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