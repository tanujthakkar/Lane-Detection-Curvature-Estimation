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

        histograms = np.zeros([3, 256], dtype=int)
        bins = np.arange(256)

        b, g, r = cv2.split(img)
        img_split = [b.flatten(), g.flatten(), r.flatten()]

        for c, channel in enumerate(img_split):
            for pix in channel:
                histograms[c][pix] += 1

        plt.figure()
        plt.title("Histograms")
        plt.xlabel("Pixel Value")
        plt.ylabel("Pixel Count")
        plt.xlim([0, 255])
        plt.plot(bins, histograms[0], color='b')
        plt.plot(bins, histograms[1], color='g')
        plt.plot(bins, histograms[2], color='r')
        plt.show()

        return histograms, img_split

    def __compute_cdf(self, histograms: np.array, img_split: list) -> np.array:
        '''
            Compute cumulative distribution function (CDF) of a histogram

            Input(s):
            histogram: Input histogram

            Output(s):
            cdf: CDF of input histogram
        '''

        cdf = np.zeros([3, 256], dtype=int)
        cdf[:, 0] = histograms[:, 0]

        for channel in range(len(histograms)):
            for _bin in range(1, len(histograms[0])):
                cdf[channel][_bin] = histograms[channel][_bin] + cdf[channel][_bin-1]

        return cdf

    def equalize(self) -> np.array:
        histogram, img_split = self.__compute_histogram(self.img)
        cdf = self.__compute_cdf(histogram, img_split)
        cdf_min = np.min(cdf[np.nonzero(cdf)])
        print(cdf_min)

        # output = np.zeros(self.img_gray.flatten().shape, dtype=int)

        # h, w = self.img_gray.shape

        # for i, pixel in enumerate(self.img_gray.flatten()):
        #     output[i] = ((cdf[pixel] - cdf_min)/((h*w) - cdf_min)) * 255

        # output = output.reshape(self.img_gray.shape)
        # print(output.shape)

        # self.output = output

        # return output

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