//
// Sort.h: SORT(Simple Online and Realtime Tracking) Class Declaration
//

#ifndef SORT_H
#define SORT_H

#include <vector>
#include <set>
#include <iomanip> // to format image names using setw() and setfill()

#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "Hungarian.h"
#include "KalmanTracker.h"

//using namespace std;
//using namespace cv;


// definition of a tracking bbox
typedef struct TrackingBox
{
    int id;
    cv::Rect_<float> box;
}TrackingBox;


// This class represents the internel state of individual tracked objects observed as bounding box.
class Sort
{
public:
    Sort()
    {
        m_max_age = 5;
        m_min_hits = 3;
        m_iou_threshold = 0.3;
        m_frame_count = 0;
    }

    Sort(int max_age, int min_hits, double iou_threshold)
        : m_max_age(max_age),
          m_min_hits(min_hits),
          m_iou_threshold(iou_threshold)
    {
        m_frame_count = 0;
    }

    ~Sort() {}

    void update(std::vector<TrackingBox>& detFrameData);

    std::vector<TrackingBox> m_tracking_output;

private:
    int m_max_age;
    int m_min_hits;
    double m_iou_threshold;
    int m_frame_count;
    std::vector<KalmanTracker> m_trackers;
};

#endif  //SORT_H
