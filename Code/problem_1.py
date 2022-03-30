#!/usr/bin/python3

# Importing modules
import os
import sys
import numpy as np
import cv2



class HistogramEqulization:

    def __init__(self, img_path):
        self.img = cv2.imread(img_path)
        self.img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.output = None

    


    def visualize(self):
        cv2.imshow("Image", self.img)
        cv2.imshow("Image Grayscale", self.img_gray)
        cv2.waitKey()


def main():

    HE = HistogramEqulization('../Data/adaptive_hist_data/0000000000.png')
    HE.visualize()


if __name__ == '__main__':
    main()