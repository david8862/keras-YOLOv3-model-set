// --------------------------------------------------------------------
// Kalman Filter Demonstrating, a 2-D mouse tracking ball demo
//
// Reference doc:
// https://blog.csdn.net/GDFSG/article/details/50904811
// --------------------------------------------------------------------
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


void kalmanTest(void)
{
    // define state value (x, y, x', y') number, measurement value (x, y) number,
    // here x' is speed of x
    int stateNum = 4;
    int measureNum = 2;

    // create kalman filter object, 0 is control value number
    cv::KalmanFilter kalman = cv::KalmanFilter(stateNum, measureNum, 0);

    // initialize measurement values with 0
    cv::Mat measurement = cv::Mat::zeros(measureNum, 1, CV_32F);

    // define state transition matrix, here we assume
    // uniform linear motion for both x and y:
    // x*  = x + t * x' (x* is x for next step; t = 1)
    // y*  = y + t * y'
    // x*' = x'
    // y*' = y'
    //
    // in form of matrix:
    //
    // / x* \ = / 1 0 1 0 \ / x \
    // | y* | = | 0 1 0 1 | | y |
    // | x*'| = | 0 0 1 0 | | x'|
    // \ y*'/ = \ 0 0 0 1 / \ y'/
    //
    kalman.transitionMatrix = (cv::Mat_<float>(stateNum, stateNum) <<
            1, 0, 1, 0,
            0, 1, 0, 1,
            0, 0, 1, 0,
            0, 0, 0, 1);

    // initialize measurement matrix with diag(1)
    cv::setIdentity(kalman.measurementMatrix, cv::Scalar::all(1));
    // initialize system noise matrix with diag(0.01)
    cv::setIdentity(kalman.processNoiseCov, cv::Scalar::all(1e-2));
    // initialize measurement noise matrix with diag(0.1)
    cv::setIdentity(kalman.measurementNoiseCov, cv::Scalar::all(1e-1));
    // initialize minimum mean squared error matrix with diag(1)
    cv::setIdentity(kalman.errorCovPost, cv::Scalar::all(1));

    // initialize state value to random
    randn(kalman.statePost, cv::Scalar::all(0), cv::Scalar::all(winHeight));

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

        // pick measurement value from mouse position
        cv::Point statePt = mousePosition;
        measurement.at<float>(0, 0) = statePt.x;
        measurement.at<float>(1, 0) = statePt.y;

        // update measurement value
        kalman.correct(measurement);

        // visualization predict & current point
        img.setTo(cv::Scalar(255, 255, 255));
        cv::circle(img, predictPt, 8, CV_RGB(0, 255, 0), -1); // predicted point as green
        cv::circle(img, statePt, 8, CV_RGB(255, 0, 0), -1); // current position as red

        cv::imshow("Kalman", img);
        char code = (char)cv::waitKey(100);
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
