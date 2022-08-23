// --------------------------------------------------------------------
// Kalman Filter Demonstrating, a 2-D mouse tracking ball demo
//
// Reference doc:
// https://blog.csdn.net/GDFSG/article/details/50904811
// --------------------------------------------------------------------
#include <random>
#include <stdio.h>
#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"

//using namespace std;
//using namespace cv;

// UI window size
const int winHeight = 600;
const int winWidth = 800;

// use a global data to track mouse position
cv::Point mousePosition = cv::Point(winWidth >> 1, winHeight >> 1);

// mouse event callback to catch mouse position from system
void mouseEvent(int event, int x, int y, int flags, void *param)
{
    if (event == CV_EVENT_MOUSEMOVE) {
        mousePosition = cv::Point(x, y);
    }
}


int randomIntGenerate(const int low, const int high)
{
    static std::random_device rd;
    static std::default_random_engine eng(rd());
    //static std::mt19937 eng(rd());

    std::uniform_int_distribution<int> dis(low, high);

    return dis(eng);
}


float randomFloatGenerate(const float low, const float high)
{
    static std::random_device rd;
    static std::default_random_engine eng(rd());
    //static std::mt19937 eng(rd());

    std::uniform_real_distribution<float> dis(low, high);
    //std::normal_distribution<float> dis((high+low)/2, (high-low)/2);

    return dis(eng);
}

void kalmanTest(void)
{
    // define state value (x, y, x', y') number, measurement value (x, y) number,
    // here x', y' is speed of x, y
    const int stateNum = 4;
    const int measureNum = 2;
    const int controlNum = 0;
    const float dt = 0.1;

    // create kalman filter object, 0 is control value number
    cv::KalmanFilter kalman = cv::KalmanFilter(stateNum, measureNum, controlNum);

    ////////////////////////////////////////////////////////////////
    // Kalman filter param config

    // define state transition matrix, here we assume
    // uniform linear motion for both x and y:
    // x*  = x + dt * x' (x* is x for next step; dt = 0.1s)
    // y*  = y + dt * y'
    // x*' = x'
    // y*' = y'
    //
    // in form of matrix:
    //
    // / x* \   / 1  0 dt  0 \ / x \
    // | y* | = | 0  1  0 dt | | y |
    // | x*'|   | 0  0  1  0 | | x'|
    // \ y*'/   \ 0  0  0  1 / \ y'/
    //
    kalman.transitionMatrix = (cv::Mat_<float>(stateNum, stateNum) <<
            1,  0, dt,  0,
            0,  1,  0, dt,
            0,  0,  1,  0,
            0,  0,  0,  1);

    // define measurement matrix. since state value is (x, y, x', y')
    // and measurement value is (x, y), measurement would be:
    // x* = x
    // y* = y
    //
    // in form of matrix:
    //
    //                         / x \
    // / x* \ = / 1  0  0  0 \ | y |
    // \ y* /   \ 0  1  0  0 / | x'|
    //                         \ y'/
    //
    cv::setIdentity(kalman.measurementMatrix, cv::Scalar::all(1));

    // set process noise covariance matrix with diag(0.01)
    cv::setIdentity(kalman.processNoiseCov, cv::Scalar::all(1e-2));

    // set measurement noise covariance matrix with diag(0.1)
    cv::setIdentity(kalman.measurementNoiseCov, cv::Scalar::all(1e-1));

    // initialize error estimate covariance matrix with diag(1),
    // which would be update during following loop
    cv::setIdentity(kalman.errorCovPost, cv::Scalar::all(1));

    // initialize state value with random
    randn(kalman.statePost, cv::Scalar::all(0), cv::Scalar::all(winHeight));

    // initialize measurement values with 0
    cv::Mat measurement = cv::Mat::zeros(measureNum, 1, CV_32F);
    ////////////////////////////////////////////////////////////////


    // prepare UI window
    cv::namedWindow("Kalman");
    cv::setMouseCallback("Kalman", mouseEvent);
    cv::Mat img(winHeight, winWidth, CV_8UC3);

    // predict-correct loop
    while (1)
    {
        // get predict value and convert to point coordinate
        cv::Mat prediction = kalman.predict();
        cv::Point predictPt = cv::Point(prediction.at<float>(0, 0), prediction.at<float>(1, 0));

        // pick measurement value from mouse position, here we add
        // random number to simulate measurement noise
        cv::Point statePt = mousePosition;
        //measurement.at<float>(0, 0) = statePt.x + randomIntGenerate(-5, 5);
        //measurement.at<float>(1, 0) = statePt.y + randomIntGenerate(-5, 5);
        measurement.at<float>(0, 0) = statePt.x + randomFloatGenerate(-5.0, 5.0);
        measurement.at<float>(1, 0) = statePt.y + randomFloatGenerate(-5.0, 5.0);

        // update measurement value
        kalman.correct(measurement);

        // visualization predict & current point
        img.setTo(cv::Scalar(255, 255, 255));
        cv::circle(img, predictPt, 8, CV_RGB(0, 255, 0), -1); // predicted point as green
        cv::circle(img, statePt, 8, CV_RGB(255, 0, 0), -1); // current position as red

        // show predict & current point coordinate
        char buf[256];
        sprintf(buf, "predicted position:(%3d,%3d)", predictPt.x, predictPt.y);
        cv::putText(img, buf, cv::Point(10,30), CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255,0,0), 1, 8);
        sprintf(buf, "current position:(%3d,%3d)", statePt.x, statePt.y);
        cv::putText(img, buf, cv::Point(10,60), CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255,0,0), 1, 8);

        cv::imshow("Kalman", img);
        char code = (char)cv::waitKey(dt*1000);
        if (code == 27 || code == 'q' || code == 'Q')
            break;
    }
    cv::destroyWindow("Kalman");
}

int main(void)
{
    kalmanTest();
    return 0;
}
