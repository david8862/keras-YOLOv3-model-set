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
mouse_position = np.empty((2, 1), np.float32)


# mouse event callback to catch mouse position from system
def mouse_event(k, x, y, s, p):
    global mouse_position
    mouse_position = np.array([[np.float32(x)], [np.float32(y)]])


def random_generate(lower, upper):
    return np.random.rand()*(upper-lower) + lower


def kalman_test():
    # define state value (x, y, x', y') number, measurement value (x, y) number,
    # here x', y' is speed of x, y
    state_num = 4
    measure_num = 2
    control_num = 0;
    dt = 0.1

    # create kalman filter object, 0 is control value number
    kalman = cv2.KalmanFilter(state_num, measure_num, control_num)

    #############################################################
    # Kalman filter param config

    # define state transition matrix, here we assume
    # uniform linear motion for both x and y:
    # x*  = x + dt * x' (x* is x for next step; dt = 0.1s)
    # y*  = y + dt * y'
    # x*' = x'
    # y*' = y'
    #
    # in form of matrix:
    #
    # / x* \   / 1  0 dt  0 \ / x \
    # | y* | = | 0  1  0 dt | | y |
    # | x*'|   | 0  0  1  0 | | x'|
    # \ y*'/   \ 0  0  0  1 / \ y'/
    kalman.transitionMatrix = np.array([[1,  0, dt,  0],
                                        [0,  1,  0, dt],
                                        [0,  0,  1,  0],
                                        [0,  0,  0,  1]], dtype=np.float32)

    # define measurement matrix. since state value is (x, y, x', y')
    # and measurement value is (x, y), measurement would be:
    # x* = x
    # y* = y
    #
    # in form of matrix:
    #
    #                         / x \
    # / x* \ = / 1  0  0  0 \ | y |
    # \ y* /   \ 0  1  0  0 / | x'|
    #                         \ y'/
    #
    kalman.measurementMatrix = np.eye(measure_num, state_num, dtype=np.float32)

    # set process noise covariance matrix with diag(0.01)
    kalman.processNoiseCov = np.eye(state_num, dtype=np.float32) * 0.01

    # set measurement noise covariance matrix with diag(0.1)
    kalman.measurementNoiseCov = np.eye(measure_num, dtype=np.float32) * 0.1

    # initialize error estimate covariance matrix with diag(1),
    # which would be update during following loop
    kalman.errorCovPost = np.eye(state_num, dtype=np.float32)

    # initialize state value with 0
    kalman.statePost = np.zeros(state_num, dtype=np.float32)
    #############################################################


    # prepare UI window
    cv2.namedWindow("Kalman")
    cv2.setMouseCallback("Kalman", mouse_event)

    # predict-correct loop
    while True:
        # get predict value
        predict_point = kalman.predict()

        # pick measurement value from mouse position, here we add
        # random number to simulate measurement noise
        state_point = mouse_position
        random_noise = np.array([[random_generate(-5.0, 5.0)], [random_generate(-5.0, 5.0)]], dtype=np.float32)
        measurement = state_point + random_noise

        # update measurement value with mouse position
        kalman.correct(measurement)

        # visualization predict & current point
        image = np.ones((600, 800, 3), np.uint8) * 255 # drawing canvas
        cv2.circle(image, center=(int(predict_point[0]), int(predict_point[1])), color=(0, 255, 0), radius=8, thickness=-1) # predicted point as green
        cv2.circle(image, center=(int(mouse_position[0]), int(mouse_position[1])), color=(0, 0, 255), radius=8, thickness=-1) # current position as red

        # show predict & current point coordinate
        cv2.putText(image, "predicted position:(%d,%d)" % (int(predict_point[0]), int(predict_point[1])), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
        cv2.putText(image, "current position:(%d,%d)" % (int(mouse_position[0]), int(mouse_position[1])), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)

        cv2.imshow("Kalman", image)

        code = cv2.waitKey(int(dt*1000)) & 0xFF
        if code == 27 or code == ord('q') or code == ord('Q'):
            break

    cv2.destroyWindow("Kalman")



def main():
    kalman_test()


if __name__ == '__main__':
    main()
