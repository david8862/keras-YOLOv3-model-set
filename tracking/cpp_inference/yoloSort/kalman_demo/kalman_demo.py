#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kalman Filter Demonstrating, a 2-D mouse tracking ball demo

Reference code & doc:
https://www.coder.work/article/95666
https://blog.csdn.net/GDFSG/article/details/50904811
"""
import os, sys, argparse
import cv2
import numpy as np

# use a global data to track mouse position
mouse_position = np.array((2,1), np.float32)


# mouse event callback to catch mouse position from system
def mouse_event(k, x, y, s, p):
    global mouse_position
    mouse_position = np.array([[np.float32(x)],[np.float32(y)]])


def kalman_test():
    # define state value (x, y, x', y') number, measurement value (x, y) number,
    # here x' is speed of x
    state_num = 4
    measure_num = 2

    # create kalman filter object, 0 is control value number
    kalman = cv2.KalmanFilter(state_num, measure_num, 0)

    # define state transition matrix, here we assume
    # uniform linear motion for both x and y:
    # x*  = x + t * x' (x* is x for next step; t = 1)
    # y*  = y + t * y'
    # x*' = x'
    # y*' = y'
    #
    # in form of matrix:
    #
    # / x* \ = / 1 0 1 0 \ / x \
    # | y* | = | 0 1 0 1 | | y |
    # | x*'| = | 0 0 1 0 | | x'|
    # \ y*'/ = \ 0 0 0 1 / \ y'/
    #
    kalman.transitionMatrix = np.array([[1,0,1,0],
                                        [0,1,0,1],
                                        [0,0,1,0],
                                        [0,0,0,1]], dtype=np.float32)

    # initialize measurement matrix with diag(1)
    kalman.measurementMatrix = np.eye(measure_num, state_num, dtype=np.float32)
    # initialize system noise matrix with diag(0.01)
    kalman.processNoiseCov = np.eye(state_num, dtype=np.float32) * 0.01
    # initialize measurement noise matrix with diag(0.1)
    kalman.measurementNoiseCov = np.eye(measure_num, dtype=np.float32) * 0.1
    # initialize minimum mean squared error matrix with diag(1)
    kalman.errorCovPost = np.eye(state_num, dtype=np.float32)

    # prepare UI window
    cv2.namedWindow("Kalman")
    cv2.setMouseCallback("Kalman", mouse_event)

    # predict-correct loop
    while True:
        # get predict value
        predict_point = kalman.predict()

        # update measurement value with mouse position
        kalman.correct(mouse_position)

        # visualization predict & current point
        frame = np.ones((600, 800, 3), np.uint8) * 255 # drawing canvas
        cv2.circle(frame, center=(int(predict_point[0]), int(predict_point[1])), color=(0, 255, 0), radius=8, thickness=-1) # predicted point as green
        cv2.circle(frame, center=(int(mouse_position[0]), int(mouse_position[1])), color=(0, 0, 255), radius=8, thickness=-1) # current position as red

        cv2.imshow("Kalman",frame)

        code = cv2.waitKey(100) & 0xFF
        if code == 27 or code == 'q' or code == 'Q':
            break

    cv2.destroyWindow("Kalman")



def main():
    kalman_test()


if __name__ == '__main__':
    main()
