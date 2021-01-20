//
// Sort.cpp: SORT(Simple Online and Realtime Tracking) Class Implementation
//
#include "Sort.h"


// Computes IOU between two bounding boxes
double get_iou(cv::Rect_<float> bb_test, cv::Rect_<float> bb_gt)
{
    float in = (bb_test & bb_gt).area();
    float un = bb_test.area() + bb_gt.area() - in;

    if (un < DBL_EPSILON)
        return 0;

    return (double)(in / un);
}

// Update the state vector with observed bounding box.
void Sort::update(std::vector<TrackingBox>& detect_frame_data)
{
    m_frame_count += 1;

    if (m_trackers.size() == 0) { // the first frame met
        // initialize kalman trackers using first detections.
        for (unsigned int i = 0; i < detect_frame_data.size(); i++) {
            KalmanTracker track = KalmanTracker(detect_frame_data[i].box);
            m_trackers.push_back(track);
        }
        return;
        //continue;
    }

    ///////////////////////////////////////
    // 1. get predicted locations from existing trackers.
    std::vector<cv::Rect_<float>> predicted_boxes;
    for (auto it = m_trackers.begin(); it != m_trackers.end();) {
        cv::Rect_<float> pBox = (*it).predict();
        if (pBox.x >= 0 && pBox.y >= 0) {
            predicted_boxes.push_back(pBox);
            it++;
        } else {
            it = m_trackers.erase(it);
        }
    }

    ///////////////////////////////////////
    // 2. associate detections to tracked object (both represented as bounding boxes)
    unsigned int track_num = predicted_boxes.size();
    unsigned int detect_num = detect_frame_data.size();

    std::vector<std::vector<double>> iou_matrix;
    iou_matrix.resize(track_num, std::vector<double>(detect_num, 0));

    for (unsigned int i = 0; i < track_num; i++) { // compute iou matrix as a distance matrix
        for (unsigned int j = 0; j < detect_num; j++) {
            // use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
            iou_matrix[i][j] = 1 - get_iou(predicted_boxes[i], detect_frame_data[j].box);
        }
    }

    // solve the assignment problem using hungarian algorithm.
    // the resulting assignment is [track(prediction) : detection], with len=preNum
    std::vector<int> assignment;
    HungarianAlgorithm HungAlgo;
    HungAlgo.Solve(iou_matrix, assignment);

    // find matches, unmatched_detections and unmatched_predictions
    std::set<int> unmatched_detections;
    std::set<int> unmatched_trajectories;
    std::set<int> all_items;
    std::set<int> matched_items;

    if (detect_num > track_num) { // there are unmatched detections
        for (unsigned int n = 0; n < detect_num; n++)
            all_items.insert(n);

        for (unsigned int i = 0; i < track_num; ++i)
            matched_items.insert(assignment[i]);

        std::set_difference(all_items.begin(), all_items.end(),
                matched_items.begin(), matched_items.end(),
                std::insert_iterator<std::set<int>>(unmatched_detections, unmatched_detections.begin()));
    }
    else if (detect_num < track_num) { // there are unmatched trajectory/predictions
        for (unsigned int i = 0; i < track_num; ++i)
            if (assignment[i] == -1) // unassigned label will be set as -1 in the assignment algorithm
                unmatched_trajectories.insert(i);
    }

    // filter out matched with low IOU
    std::vector<cv::Point> matched_pairs;
    for (unsigned int i = 0; i < track_num; ++i) {
        if (assignment[i] == -1) { // pass over invalid values
            continue;
        }
        if (1 - iou_matrix[i][assignment[i]] < m_iou_threshold) {
            unmatched_trajectories.insert(i);
            unmatched_detections.insert(assignment[i]);
        }
        else
            matched_pairs.push_back(cv::Point(i, assignment[i]));
    }

    ///////////////////////////////////////
    // 3. updating trackers
    // update matched trackers with assigned detections.
    // each prediction is corresponding to a tracker
    int detect_index, track_index;
    for (unsigned int i = 0; i < matched_pairs.size(); i++) {
        track_index = matched_pairs[i].x;
        detect_index = matched_pairs[i].y;
        m_trackers[track_index].update(detect_frame_data[detect_index].box);
    }

    // create and initialise new trackers for unmatched detections
    for (auto unmatched_det_index : unmatched_detections) {
        KalmanTracker tracker = KalmanTracker(detect_frame_data[unmatched_det_index].box);
        m_trackers.push_back(tracker);
    }

    // get trackers' output
    m_tracking_output.clear();
    for (auto it = m_trackers.begin(); it != m_trackers.end();) {
        if (((*it).m_time_since_update < 1) &&
                ((*it).m_hit_streak >= m_min_hits || m_frame_count <= m_min_hits)) {
            TrackingBox output;
            output.box = (*it).get_state();
            output.id = (*it).m_id + 1;
            m_tracking_output.push_back(output);
            it++;
        }
        else
            it++;

        // remove dead tracklet
        if (it != m_trackers.end() && (*it).m_time_since_update > m_max_age)
            it = m_trackers.erase(it);
    }

    return;
}

