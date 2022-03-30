#!/usr/env/bin python3

"""
ENPM673 Spring 2022: Perception for Autonomous Robots

Project 2 - Lane Detection & Curvature Estimation

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


class HistogramEqulization:

    def __init__(self, img: np.array) -> None:
        self.img = img
        self.img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.output = None

    def __compute_histogram(self, img: np.array) -> np.array:
        '''
            Compute the histogram of an image

            Input(s):
            img: Input image

            Output(s):
            histogram: Histogram of input image
        '''

        histogram = np.zeros(256, dtype=int)
        bins = np.arange(256)

        for i in img.flatten():
            histogram[i] += 1

        plt.figure()
        plt.title("Grayscale Histogram")
        plt.xlabel("Pixel Value")
        plt.ylabel("Pixel Count")
        plt.xlim([0, 255])
        plt.plot(bins, histogram)
        # plt.show()

        return histogram

    def __compute_cdf(self, histogram: np.array) -> np.array:
        '''
            Compute cumulative distribution function (CDF) of a histogram

            Input(s):
            histogram: Input histogram

            Output(s):
            cdf: CDF of input histogram
        '''

        cdf = np.zeros(256, dtype=int)
        cdf[0] = histogram[0]

        for _bin in range(1, len(histogram)):
            cdf[_bin] = histogram[_bin] + cdf[_bin-1]

        return cdf

    def equalize(self, img: np.array) -> np.array:

        # b, g, r = cv2.split(img)
        # img_split = [b, g, r]

        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(img)
        img_split = [h, s, v]

        output = list()

        for i, c in enumerate(img_split[:1]):
            histogram = self.__compute_histogram(c)
            cdf = self.__compute_cdf(histogram)
            cdf_min = np.min(cdf[np.nonzero(cdf)])

            channel = np.zeros(c.flatten().shape, dtype=int)

            h, w = c.shape

            for i, pixel in enumerate(c.flatten()):
                channel[i] = ((cdf[pixel] - cdf_min)/((h*w) - cdf_min)) * 255

            channel = channel.reshape(c.shape)
            output.append(channel)

        output = np.dstack((channel, s, v))

        # self.output = output
        # cv2.imshow("Result", np.uint8(output))
        # cv2.waitKey()

        return output

    def adaptive_equalize(self) -> np.array:

        self.output = np.copy(self.img)

        r_step = self.img.shape[0]//8
        c_step = self.img.shape[1]//8

        for r in range(0, self.img.shape[0], r_step):
            for c in range(0, self.img.shape[1], c_step):
                self.output[r:r+r_step, c:c+c_step, :] = self.equalize(self.img[r:r+r_step, c:c+c_step, :])

    def visualize(self) -> None:
        cv2.imshow("Image", self.img)
        cv2.imshow("Result", np.uint8(self.output))
        cv2.waitKey()


def hist_equalize_img(img_path: str) -> np.array:

    img = cv2.imread(img_path)
    HE = HistogramEqulization(img)
    # HE.equalize(img)
    HE.adaptive_equalize()
    HE.visualize()


def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ImagePath', type=str, default="../Data/adaptive_hist_data/0000000024.png", help='Path to input image')
    Parser.add_argument('--Visualize', action='store_true', help='Toggle visualization')
    
    Args = Parser.parse_args()
    image_path = Args.ImagePath

    hist_equalize_img(image_path)    


if __name__ == '__main__':
    main()