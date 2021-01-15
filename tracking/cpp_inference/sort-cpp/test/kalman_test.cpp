// --------------------------------------------------------------------
// Kalman Filter Demonstrating, a 2-d ball demo
// --------------------------------------------------------------------
#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"

//using namespace std;
//using namespace cv;


const int winHeight = 600;
const int winWidth = 800;

cv::Point mousePosition = cv::Point(winWidth >> 1, winHeight >> 1);

// mouse event callback
void mouseEvent(int event, int x, int y, int flags, void *param)
{
    if (event == CV_EVENT_MOUSEMOVE) {
        mousePosition = cv::Point(x, y);
    }
}


void kalmanTest(void)
{
    int stateNum = 4;
    int measureNum = 2;
    cv::KalmanFilter kf = cv::KalmanFilter(stateNum, measureNum, 0);

    // initialization
    cv::Mat processNoise(stateNum, 1, CV_32F);
    cv::Mat measurement = cv::Mat::zeros(measureNum, 1, CV_32F);

    kf.transitionMatrix = (cv::Mat_<float>(stateNum, stateNum) <<
            1, 0, 1, 0,
            0, 1, 0, 1,
            0, 0, 1, 0,
            0, 0, 0, 1);

    cv::setIdentity(kf.measurementMatrix);
    cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-2));
    cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1e-1));
    cv::setIdentity(kf.errorCovPost, cv::Scalar::all(1));

    randn(kf.statePost, cv::Scalar::all(0), cv::Scalar::all(winHeight));

    cv::namedWindow("Kalman");
    cv::setMouseCallback("Kalman", mouseEvent);
    cv::Mat img(winHeight, winWidth, CV_8UC3);

    while (1)
    {
        // predict
        cv::Mat prediction = kf.predict();
        cv::Point predictPt = cv::Point(prediction.at<float>(0, 0), prediction.at<float>(1, 0));

        // generate measurement
        cv::Point statePt = mousePosition;
        measurement.at<float>(0, 0) = statePt.x;
        measurement.at<float>(1, 0) = statePt.y;

        // update
        kf.correct(measurement);

        // visualization
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
