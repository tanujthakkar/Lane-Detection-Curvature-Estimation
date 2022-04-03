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

    def __compute_histogram(self, img: np.array, clip: int = 0) -> np.array:
        '''
            Compute the histogram of an image

            Input(s):
            img: Input image

            Output(s):
            histogram: Histogram of input image
        '''

        histogram = np.zeros(256, dtype=int)
        bins = np.arange(256)
        residue = 0

        for i in img.flatten():
            if(clip > 0 and histogram[i] >= clip):
                    residue += 1
            else:
                histogram[i] += 1

        if(clip):
            histogram += (residue//256)

        # plt.figure()
        # plt.title("Grayscale Histogram")
        # plt.xlabel("Pixel Value")
        # plt.ylabel("Pixel Count")
        # plt.xlim([0, 255])
        # plt.plot(bins, histogram)
        # plt.show()

        # print(histogram)

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

    def equalize(self, img: np.array, clip: int = 0) -> np.array:

        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_img)
        img_split = [h, s, v]

        output = list()

        c = v

        histogram = self.__compute_histogram(c, clip)
        cdf = self.__compute_cdf(histogram)
        cdf_min = np.min(cdf[np.nonzero(cdf)])

        channel = np.zeros(c.flatten().shape, dtype=int)

        h, w = c.shape

        for i, pixel in enumerate(c.flatten()):
            channel[i] = ((cdf[pixel] - cdf_min)/((h*w) - cdf_min)) * 255

        channel = channel.reshape(c.shape)
        output.append(channel)

        hsv_img[:,:,2] = channel
        output = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

        return output

    def adaptive_equalize(self, img: np.array, tile_size: tuple = (64, 64), clip: int = 0) -> np.array:

        self.output = np.copy(img)

        r_step, c_step = tile_size

        for r in range(0, img.shape[0], r_step):
            for c in range(0, img.shape[1], c_step):
                self.output[r:r+r_step, c:c+c_step, :] = self.equalize(img[r:r+r_step, c:c+c_step, :], clip)

        return self.output


def hist_equalize_img(img_path: str, clahe: bool = False, visualize: bool = False) -> np.array:

    img = cv2.imread(img_path)
    HE = HistogramEqulization(img)
    if(not clahe):
        output = HE.equalize(img)
    else:
        output = HE.adaptive_equalize(img, clip=10)
    
    if(visualize):
        cv2.imshow("Histogram Equlization", np.vstack((img, output)))
        cv2.waitKey(5)

    return output

def hist_equalize_data(data_dir: str, result_dir: str, clahe: bool = False, visualize: bool = False) -> None:

    img_set = read_image_set(data_dir)

    if(clahe):
        result_dir = os.path.join(result_dir, data_dir.split('/')[-2] + '_processed_CLAHE')
    else:
        result_dir = os.path.join(result_dir, data_dir.split('/')[-2] + '_processed')
    if(not os.path.exists(result_dir)):
        os.makedirs(result_dir, exist_ok=True)

    for img in img_set:
        output_file = img.split('/')[-1].split('.')[0] + '.png'
        output = hist_equalize_img(img, clahe, visualize)
        cv2.imwrite(os.path.join(result_dir, output_file), output)

def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ImagePath', type=str, default=None, help='Path to input image')
    Parser.add_argument('--DataDir', type=str, default="../Data/adaptive_hist_data/", help='Path to data directory')
    Parser.add_argument('--ResultDir', type=str, default="../Result/", help='Path to results directory')
    Parser.add_argument('--CLAHE', action='store_true', help='Toggle CLAHE')
    Parser.add_argument('--Visualize', action='store_true', help='Toggle visualization')
    
    Args = Parser.parse_args()
    image_path = Args.ImagePath
    data_dir = Args.DataDir
    result_dir = Args.ResultDir
    clahe = Args.CLAHE
    visualize = Args.Visualize

    if(image_path):
        hist_equalize_img(image_path, clahe, visualize)
    else:
        hist_equalize_data(data_dir, result_dir, clahe, visualize)


if __name__ == '__main__':
    main()