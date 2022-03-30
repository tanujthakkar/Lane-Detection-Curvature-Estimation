#!/usr/bin/python3

# Importing modules
import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

from utils import *


class HistogramEqulization:

    def __init__(self, img_path: str) -> None:
        self.img = cv2.imread(img_path)
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

    def equalize(self) -> np.array:

        b, g, r = cv2.split(self.img)
        img_split = [b, g, r]
        self.output = list()

        for i, c in enumerate(img_split):
            histogram = self.__compute_histogram(c)
            cdf = self.__compute_cdf(histogram)
            cdf_min = np.min(cdf[np.nonzero(cdf)])

            channel = np.zeros(c.flatten().shape, dtype=int)

            h, w = c.shape

            for i, pixel in enumerate(c.flatten()):
                channel[i] = ((cdf[pixel] - cdf_min)/((h*w) - cdf_min)) * 255

            channel = channel.reshape(c.shape)
            self.output.append(channel)

        self.output = cv2.merge((self.output[0], self.output[1], self.output[2]))

        return self.output

    def visualize(self) -> None:
        cv2.imshow("Image", self.img)
        cv2.imshow("Image Grayscale", self.img_gray)
        cv2.imshow("Result", np.uint8(self.output))
        cv2.waitKey()


def main():

    HE = HistogramEqulization('../Data/adaptive_hist_data/0000000024.png')
    HE.equalize()
    HE.visualize()


if __name__ == '__main__':
    main()