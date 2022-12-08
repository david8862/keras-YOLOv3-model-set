//
//  yoloDetection.cpp
//  MNN
//
//  Created by Xiaobin Zhang on 2019/09/20.
//
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <algorithm>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <iostream>
#include <vector>
#include <numeric>
#include <math.h>
#include <unistd.h>
#include <getopt.h>
#include <string.h>
#include <sys/time.h>

#define MNN_OPEN_TIME_TRACE
#include "MNN/ImageProcess.hpp"
#include "MNN/Interpreter.hpp"
#include "MNN/AutoTime.hpp"
#include "MNN/ErrorCode.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

using namespace MNN;
using namespace MNN::CV;


// definition of a bbox prediction record
typedef struct prediction {
    float x;
    float y;
    float width;
    float height;
    float confidence;
    int class_index;
}t_prediction;


// model inference settings
struct Settings {
  int loop_count = 1;
  int number_of_threads = 4;
  int number_of_warmup_runs = 2;
  float conf_thrd = 0.1f;
  float input_mean = 0.0f;
  float input_std = 255.0f;
  bool elim_grid_sense = false;
  std::string model_name = "./model.mnn";
  std::string input_img_name = "./dog.jpg";
  std::string classes_file_name = "./classes.txt";
  std::string anchors_file_name = "./yolo3_anchors.txt";
  std::string result_file_name = "./result.txt";
  bool input_floating = false;
  //bool verbose = false;
  //string input_layer_type = "uint8_t";
};


float sigmoid(float x)
{
    return (1 / (1 + exp(-x)));
}


void softmax(const std::vector<float> &logits, std::vector<float> &output){
    float sum=0.0;
    output.clear();

    for(size_t i = 0; i<logits.size(); ++i) {
        output.emplace_back(exp(logits[i]));
    }
    sum = std::accumulate(output.begin(), output.end(), sum);

    for(size_t i = 0; i<output.size(); ++i) {
        output[i] /= sum;
    }
    return;
}


double get_us(struct timeval t)
{
    return (t.tv_sec * 1000000 + t.tv_usec);
}


void display_usage() {
    std::cout
        << "Usage: yoloDetection\n"
        << "--mnn_model, -m: model_name.mnn\n"
        << "--image, -i: image_name.jpg\n"
        << "--classes, -l: classes labels for the model\n"
        << "--anchors, -a: anchor values for the model\n"
        << "--conf_thrd, -n: confidence threshold for detection filter\n"
        << "--input_mean, -b: input mean\n"
        << "--input_std, -s: input standard deviation\n"
        << "--elim_grid_sense, -e: [0|1] eliminate grid sensitivity\n"
        << "--threads, -t: number of threads\n"
        << "--count, -c: loop model run for certain times\n"
        << "--warmup_runs, -w: number of warmup runs\n"
        << "--result, -r: result txt file to save detection output\n"
        //<< "--verbose, -v: [0|1] print more information\n"
        << "\n";
    return;
}


// YOLO postprocess for each prediction feature map
void yolo_postprocess(const Tensor* feature_map, const int input_width, const int input_height,
                      const int num_classes, const std::vector<std::pair<float, float>> anchors,
                      std::vector<t_prediction> &prediction_list, float conf_threshold, float scale_x_y)
{
    // 1. do following transform to get the output bbox,
    //    which is aligned with YOLOv3/YOLOv2 paper:
    //
    //    bbox_x = (sigmoid(pred_x) + grid_w) / grid_width
    //    bbox_y = (sigmoid(pred_y) + grid_h) / grid_height
    //    bbox_w = (exp(pred_w) * anchor_w) / input_width
    //    bbox_h = (exp(pred_h) * anchor_h) / input_height
    //    bbox_obj = sigmoid(pred_obj)
    //
    //    if using "Eliminate grid sensitivity", bbox_x & bbox_y
    //    will use following formula
    //
    //    bbox_x_tmp = sigmoid(bytes[bbox_x_offset]) * scale_x_y - (scale_x_y - 1) / 2;
    //    bbox_y_tmp = sigmoid(bytes[bbox_y_offset]) * scale_x_y - (scale_x_y - 1) / 2;
    //    bbox_x = (bbox_x_tmp + w) / width;
    //    bbox_y = (bbox_y_tmp + h) / height;
    //
    //
    // 2. get bbox confidence and filter with threshold:
    //
    //    if(num_classes == 1) {
    //        bbox_conf[:] = bbox_obj
    //    } else {
    //        bbox_conf[:] = sigmoid/softmax(bbox_class_score[:]) * bbox_obj
    //    }
    //    bbox_max_conf = max(bbox_conf[:])
    //    bbox_max_index = argmax(bbox_conf[:])
    //
    //
    // 3. filter bbox_max_conf with threshold
    //
    //    if(bbox_max_conf > conf_threshold)
    //        enqueue the bbox info

    const float* data = feature_map->host<float>();
    auto dimType = feature_map->getDimensionType();

    auto batch   = feature_map->batch();
    auto channel = feature_map->channel();
    auto height  = feature_map->height();
    auto width   = feature_map->width();

    int stride = input_width / width;
    auto unit = sizeof(float);
    int anchor_num_per_layer = anchors.size();

    // now we only support single image postprocess
    MNN_ASSERT(batch == 1);

    // the featuremap channel should be like 3*(num_classes + 5)
    MNN_ASSERT(anchor_num_per_layer * (num_classes + 5) == channel);

    int bytesPerRow, bytesPerImage, bytesPerBatch;
    if (dimType == Tensor::TENSORFLOW) {
        // Tensorflow format tensor, NHWC
        MNN_PRINT("Tensorflow format: NHWC\n");

        bytesPerRow   = channel * unit;
        bytesPerImage = width * bytesPerRow;
        bytesPerBatch = height * bytesPerImage;

    } else if (dimType == Tensor::CAFFE) {
        // Caffe format tensor, NCHW
        MNN_PRINT("Caffe format: NCHW\n");

        bytesPerRow   = width * unit;
        bytesPerImage = height * bytesPerRow;
        bytesPerBatch = channel * bytesPerImage;

    } else if (dimType == Tensor::CAFFE_C4) {
        MNN_PRINT("Caffe format: NC4HW4, not supported\n");
        exit(-1);
    } else {
        MNN_PRINT("Invalid tensor dim type: %d\n", dimType);
        exit(-1);
    }

    for (int b = 0; b < batch; b++) {
        auto bytes = data + b * bytesPerBatch / unit;
        MNN_PRINT("batch %d:\n", b);

        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                for (int anc = 0; anc < anchor_num_per_layer; anc++) {
                    //get bbox prediction data offset for each anchor, each feature point
                    int bbox_x_offset, bbox_y_offset, bbox_w_offset, bbox_h_offset, bbox_obj_offset, bbox_scores_offset, bbox_scores_step;
                    if (dimType == Tensor::TENSORFLOW) {
                        // Tensorflow format tensor, NHWC
                        bbox_x_offset = h * width * channel + w * channel + anc * (num_classes + 5);
                        bbox_y_offset = h * width * channel + w * channel + anc * (num_classes + 5) + 1;
                        bbox_w_offset = h * width * channel + w * channel + anc * (num_classes + 5) + 2;
                        bbox_h_offset = h * width * channel + w * channel + anc * (num_classes + 5) + 3;
                        bbox_obj_offset = h * width * channel + w * channel + anc * (num_classes + 5) + 4;
                        bbox_scores_offset = h * width * channel + w * channel + anc * (num_classes + 5) + 5;
                        bbox_scores_step = 1;

                    } else if (dimType == Tensor::CAFFE) {
                        // Caffe format tensor, NCHW
                        bbox_x_offset = anc * (num_classes + 5) * width * height + h * width + w;
                        bbox_y_offset = (anc * (num_classes + 5) + 1) * width * height + h * width + w;
                        bbox_w_offset = (anc * (num_classes + 5) + 2) * width * height + h * width + w;
                        bbox_h_offset = (anc * (num_classes + 5) + 3) * width * height + h * width + w;
                        bbox_obj_offset = (anc * (num_classes + 5) + 4) * width * height + h * width + w;
                        bbox_scores_offset = (anc * (num_classes + 5) + 5) * width * height + h * width + w;
                        bbox_scores_step = width * height;

                    } else if (dimType == Tensor::CAFFE_C4) {
                        MNN_PRINT("Caffe format: NC4HW4, not supported\n");
                        exit(-1);
                    } else {
                        MNN_PRINT("Invalid tensor dim type: %d\n", dimType);
                        exit(-1);
                    }
                    // get anchor objectness score
                    float bbox_obj = sigmoid(bytes[bbox_obj_offset]);

                    // Get softmax score for YOLOv2 prediction
                    std::vector<float> logits_bbox_score;
                    std::vector<float> bbox_score;
                    if(anchor_num_per_layer == 5) {
                        for (int i = 0; i < num_classes; i++) {
                            logits_bbox_score.emplace_back(bytes[bbox_scores_offset + i * bbox_scores_step]);
                        }
                        softmax(logits_bbox_score, bbox_score);
                    }

                    // get anchor output confidence (class_score * objectness) and filter with threshold
                    float max_conf = 0.0;
                    int max_index = -1;
                    for (int i = 0; i < num_classes; i++) {
                        float tmp_conf = 0.0;
                        if(anchor_num_per_layer == 5) {
                            // YOLOv2 use 5 anchors and softmax class scores
                            if(num_classes == 1) {
                                tmp_conf = bbox_obj;
                            }
                            else {
                                tmp_conf = bbox_score[i] * bbox_obj;
                            }
                        }
                        else {
                            // check if only 1 class for different score
                            if(num_classes == 1) {
                                tmp_conf = bbox_obj;
                            }
                            else {
                                tmp_conf = sigmoid(bytes[bbox_scores_offset + i * bbox_scores_step]) * bbox_obj;
                            }
                        }

                        if(tmp_conf > max_conf) {
                            max_conf = tmp_conf;
                            max_index = i;
                        }
                    }
                    if(max_conf >= conf_threshold) {
                        // Decode YOLO predictions
                        float bbox_x, bbox_y;

                        if(scale_x_y > 0) {
                            // Eliminate grid sensitivity trick involved in YOLOv4
                            //
                            // Reference Paper & code:
                            //     "YOLOv4: Optimal Speed and Accuracy of Object Detection"
                            //     https://arxiv.org/abs/2004.10934
                            //     https://github.com/opencv/opencv/issues/17148
                            //
                            float bbox_x_tmp = sigmoid(bytes[bbox_x_offset]) * scale_x_y - (scale_x_y - 1) / 2;
                            float bbox_y_tmp = sigmoid(bytes[bbox_y_offset]) * scale_x_y - (scale_x_y - 1) / 2;
                            bbox_x = (bbox_x_tmp + w) / width;
                            bbox_y = (bbox_y_tmp + h) / height;
                        }
                        else {
                            bbox_x = (sigmoid(bytes[bbox_x_offset]) + w) / width;
                            bbox_y = (sigmoid(bytes[bbox_y_offset]) + h) / height;
                        }

                        float bbox_w = exp(bytes[bbox_w_offset]) * anchors[anc].first / input_width;
                        float bbox_h = exp(bytes[bbox_h_offset]) * anchors[anc].second / input_height;

                        // got a valid prediction, form up data and push to result vector
                        t_prediction bbox_prediction;
                        bbox_prediction.x = bbox_x;
                        bbox_prediction.y = bbox_y;
                        bbox_prediction.width = bbox_w;
                        bbox_prediction.height = bbox_h;
                        bbox_prediction.confidence = max_conf;
                        bbox_prediction.class_index = max_index;

                        prediction_list.emplace_back(bbox_prediction);
                    }
                }
            }
        }
    }

    return;
}


/////////////////////////////////////////////////////////////////

// A fast implementation of sigmoid function on X86/ARM CPU, using
//
// 1. constant approximation when out of cut-off value
// 2. quick checking table inside cut-off range
//
// Reference page:
// https://blog.csdn.net/weixin_43327597/article/details/107196976
//
#define SIGMOID_CUTOFF_BOTTOM (-6.0f)
#define SIGMOID_CUTOFF_TOP (6.0f)
#define SIGMOID_TABLE_RANGE (1000)

static float sigmoid_table[SIGMOID_TABLE_RANGE];

// init sigmoid value table, used between
// [SIGMOID_CUTOFF_BOTTOM, SIGMOID_CUTOFF_TOP]
//
// need to be called at model init stage
static inline void sigmoid_fast_init()
{
    for (int i = 0; i < SIGMOID_TABLE_RANGE; i++) {
        sigmoid_table[i] = sigmoid(SIGMOID_CUTOFF_BOTTOM + i * (SIGMOID_CUTOFF_TOP - SIGMOID_CUTOFF_BOTTOM) / SIGMOID_TABLE_RANGE);
    }

    return;
}

static inline float sigmoid_fast(float x)
{
    //return 0.5 * (x / (1.0 + fabsf(x))) + 0.5;

    if (x <= SIGMOID_CUTOFF_BOTTOM) {
        return 0;
    }
    else if (x >= SIGMOID_CUTOFF_TOP) {
        return 1;
    }
    else {
        int index = round((x - SIGMOID_CUTOFF_BOTTOM) * SIGMOID_TABLE_RANGE / (SIGMOID_CUTOFF_TOP - SIGMOID_CUTOFF_BOTTOM));
        return sigmoid_table[index];
    }
}


// A fast implementation of exp & softmax function on X86/ARM
//
// Reference page:
// https://stackoverflow.com/questions/47025373/fastest-implementation-of-the-natural-exponential-function-using-sse
//
static inline float exp_fast(float x)
{
    union { float f; int32_t i; } reinterpreter;
    int type = 0;  //0: cubic spline (speed); 1: quartic spline (accuracy)

    reinterpreter.i = (int32_t)(12102203.0f*x) + 127*(1 << 23);
    int32_t m = (reinterpreter.i >> 7) & 0xFFFF;  // copy mantissa

    if (type == 0) {
        // cubic spline approximation,
        // empirical values for small maximum relative error (8.34e-5)
        reinterpreter.i +=
            ((((((((1277*m) >> 14) + 14825)*m) >> 14) - 79749)*m) >> 11) - 626;
    }
    else {
        // quartic spline approximation,
        // empirical values for small maximum relative error (1.21e-5)
        reinterpreter.i += (((((((((((3537*m) >> 16)
            + 13668)*m) >> 18) + 15817)*m) >> 14) - 80470)*m) >> 11);
    }
    return reinterpreter.f;
}

void softmax_fast(const std::vector<float> &logits, std::vector<float> &output){
    float sum=0.0;
    output.clear();

    for(size_t i = 0; i<logits.size(); ++i) {
        output.emplace_back(exp_fast(logits[i]));
    }
    sum = std::accumulate(output.begin(), output.end(), sum);

    for(size_t i = 0; i<output.size(); ++i) {
        output[i] /= sum;
    }
    return;
}


//
// A speed optimized YOLO postprocess implementation, including
// following tricks:
// 1. postpone bbox coordinate decode after filtering with confidence
// 2. use sigmoid_fast in score/objectness decode
// 3. use softmax_fast in YOLOv2 score decode
//
void yolo_postprocess_fast(const Tensor* feature_map, const int input_width, const int input_height,
                      const int num_classes, const std::vector<std::pair<float, float>> anchors,
                      std::vector<t_prediction> &prediction_list, float conf_threshold, float scale_x_y)
{
    // 1. do following transform to get the output bbox,
    //    which is aligned with YOLOv3/YOLOv2 paper:
    //
    //    bbox_x = (sigmoid(pred_x) + grid_w) / grid_width
    //    bbox_y = (sigmoid(pred_y) + grid_h) / grid_height
    //    bbox_w = (exp(pred_w) * anchor_w) / input_width
    //    bbox_h = (exp(pred_h) * anchor_h) / input_height
    //    bbox_obj = sigmoid(pred_obj)
    //
    //    if using "Eliminate grid sensitivity", bbox_x & bbox_y
    //    will use following formula
    //
    //    bbox_x_tmp = sigmoid(bytes[bbox_x_offset]) * scale_x_y - (scale_x_y - 1) / 2;
    //    bbox_y_tmp = sigmoid(bytes[bbox_y_offset]) * scale_x_y - (scale_x_y - 1) / 2;
    //    bbox_x = (bbox_x_tmp + w) / width;
    //    bbox_y = (bbox_y_tmp + h) / height;
    //
    //
    // 2. get bbox confidence and filter with threshold:
    //
    //    if(num_classes == 1) {
    //        bbox_conf[:] = bbox_obj
    //    } else {
    //        bbox_conf[:] = sigmoid/softmax(bbox_class_score[:]) * bbox_obj
    //    }
    //    bbox_max_conf = max(bbox_conf[:])
    //    bbox_max_index = argmax(bbox_conf[:])
    //
    //
    // 3. filter bbox_max_conf with threshold
    //
    //    if(bbox_max_conf > conf_threshold)
    //        enqueue the bbox info

    const float* data = feature_map->host<float>();
    auto dimType = feature_map->getDimensionType();

    auto batch   = feature_map->batch();
    auto channel = feature_map->channel();
    auto height  = feature_map->height();
    auto width   = feature_map->width();

    int stride = input_width / width;
    auto unit = sizeof(float);
    int anchor_num_per_layer = anchors.size();

    // now we only support single image postprocess
    MNN_ASSERT(batch == 1);

    // the featuremap channel should be like 3*(num_classes + 5)
    MNN_ASSERT(anchor_num_per_layer * (num_classes + 5) == channel);

    int bytesPerRow, bytesPerImage, bytesPerBatch;
    if (dimType == Tensor::TENSORFLOW) {
        // Tensorflow format tensor, NHWC
        MNN_PRINT("Tensorflow format: NHWC\n");

        bytesPerRow   = channel * unit;
        bytesPerImage = width * bytesPerRow;
        bytesPerBatch = height * bytesPerImage;

    } else if (dimType == Tensor::CAFFE) {
        // Caffe format tensor, NCHW
        MNN_PRINT("Caffe format: NCHW\n");

        bytesPerRow   = width * unit;
        bytesPerImage = height * bytesPerRow;
        bytesPerBatch = channel * bytesPerImage;

    } else if (dimType == Tensor::CAFFE_C4) {
        MNN_PRINT("Caffe format: NC4HW4, not supported\n");
        exit(-1);
    } else {
        MNN_PRINT("Invalid tensor dim type: %d\n", dimType);
        exit(-1);
    }

    for (int b = 0; b < batch; b++) {
        auto bytes = data + b * bytesPerBatch / unit;
        MNN_PRINT("batch %d:\n", b);

        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                for (int anc = 0; anc < anchor_num_per_layer; anc++) {
                    //
                    //check bbox score and objectness first to filter invalid prediction
                    //
                    int bbox_obj_offset, bbox_scores_offset, bbox_scores_step;
                    if (dimType == Tensor::TENSORFLOW) {
                        // Tensorflow format tensor, NHWC
                        bbox_obj_offset = h * width * channel + w * channel + anc * (num_classes + 5) + 4;
                        bbox_scores_offset = h * width * channel + w * channel + anc * (num_classes + 5) + 5;
                        bbox_scores_step = 1;

                    } else if (dimType == Tensor::CAFFE) {
                        // Caffe format tensor, NCHW
                        bbox_obj_offset = (anc * (num_classes + 5) + 4) * width * height + h * width + w;
                        bbox_scores_offset = (anc * (num_classes + 5) + 5) * width * height + h * width + w;
                        bbox_scores_step = width * height;

                    } else if (dimType == Tensor::CAFFE_C4) {
                        MNN_PRINT("Caffe format: NC4HW4, not supported\n");
                        exit(-1);
                    } else {
                        MNN_PRINT("Invalid tensor dim type: %d\n", dimType);
                        exit(-1);
                    }
                    float bbox_obj = sigmoid_fast(bytes[bbox_obj_offset]);

                    // Get softmax score for YOLOv2 prediction
                    std::vector<float> logits_bbox_score;
                    std::vector<float> bbox_score;
                    if(anchor_num_per_layer == 5) {
                        for (int i = 0; i < num_classes; i++) {
                            logits_bbox_score.emplace_back(bytes[bbox_scores_offset + i * bbox_scores_step]);
                        }
                        softmax_fast(logits_bbox_score, bbox_score);
                    }

                    //get anchor output confidence (class_score * objectness) and filter with threshold
                    float max_conf = 0.0;
                    int max_index = -1;
                    for (int i = 0; i < num_classes; i++) {
                        float tmp_conf = 0.0;
                        if(anchor_num_per_layer == 5) {
                            // YOLOv2 use 5 anchors and softmax class scores
                            if(num_classes == 1) {
                                tmp_conf = bbox_obj;
                            }
                            else {
                                tmp_conf = bbox_score[i] * bbox_obj;
                            }
                        }
                        else {
                            // check if only 1 class for different score
                            if(num_classes == 1) {
                                tmp_conf = bbox_obj;
                            }
                            else {
                                tmp_conf = sigmoid(bytes[bbox_scores_offset + i * bbox_scores_step]) * bbox_obj;
                            }
                        }

                        if(tmp_conf > max_conf) {
                            max_conf = tmp_conf;
                            max_index = i;
                        }
                    }
                    if(max_conf >= conf_threshold) {
                        // got a valid prediction, decode bbox and form up data to push to result vector
                        int bbox_x_offset, bbox_y_offset, bbox_w_offset, bbox_h_offset;

                        if (dimType == Tensor::TENSORFLOW) {
                            // Tensorflow format tensor, NHWC
                            bbox_x_offset = h * width * channel + w * channel + anc * (num_classes + 5);
                            bbox_y_offset = h * width * channel + w * channel + anc * (num_classes + 5) + 1;
                            bbox_w_offset = h * width * channel + w * channel + anc * (num_classes + 5) + 2;
                            bbox_h_offset = h * width * channel + w * channel + anc * (num_classes + 5) + 3;

                        } else if (dimType == Tensor::CAFFE) {
                            // Caffe format tensor, NCHW
                            bbox_x_offset = anc * (num_classes + 5) * width * height + h * width + w;
                            bbox_y_offset = (anc * (num_classes + 5) + 1) * width * height + h * width + w;
                            bbox_w_offset = (anc * (num_classes + 5) + 2) * width * height + h * width + w;
                            bbox_h_offset = (anc * (num_classes + 5) + 3) * width * height + h * width + w;

                        } else if (dimType == Tensor::CAFFE_C4) {
                            MNN_PRINT("Caffe format: NC4HW4, not supported\n");
                            exit(-1);
                        } else {
                            MNN_PRINT("Invalid tensor dim type: %d\n", dimType);
                            exit(-1);
                        }

                        // Decode YOLO bbox predictions
                        float bbox_x, bbox_y;

                        if(scale_x_y > 0) {
                            // Eliminate grid sensitivity trick involved in YOLOv4
                            //
                            // Reference Paper & code:
                            //     "YOLOv4: Optimal Speed and Accuracy of Object Detection"
                            //     https://arxiv.org/abs/2004.10934
                            //     https://github.com/opencv/opencv/issues/17148
                            //
                            float bbox_x_tmp = sigmoid(bytes[bbox_x_offset]) * scale_x_y - (scale_x_y - 1) / 2;
                            float bbox_y_tmp = sigmoid(bytes[bbox_y_offset]) * scale_x_y - (scale_x_y - 1) / 2;
                            bbox_x = (bbox_x_tmp + w) / width;
                            bbox_y = (bbox_y_tmp + h) / height;
                        }
                        else {
                            bbox_x = (sigmoid(bytes[bbox_x_offset]) + w) / width;
                            bbox_y = (sigmoid(bytes[bbox_y_offset]) + h) / height;
                        }
                        float bbox_w = exp(bytes[bbox_w_offset]) * anchors[anc].first / input_width;
                        float bbox_h = exp(bytes[bbox_h_offset]) * anchors[anc].second / input_height;

                        t_prediction bbox_prediction;
                        bbox_prediction.x = bbox_x;
                        bbox_prediction.y = bbox_y;
                        bbox_prediction.width = bbox_w;
                        bbox_prediction.height = bbox_h;
                        bbox_prediction.confidence = max_conf;
                        bbox_prediction.class_index = max_index;

                        prediction_list.emplace_back(bbox_prediction);
                    }
                }
            }
        }
    }

    return;
}
//////////////////////////////////////////////////////////



//calculate IoU for 2 prediction boxes
float get_iou(t_prediction pred1, t_prediction pred2)
{
    // area for box 1
    float x1min = pred1.x;
    float x1max = pred1.x + pred1.width;
    float y1min = pred1.y;
    float y1max = pred1.y + pred1.height;
    float area1 = pred1.width * pred1.height;

    // area for box 2
    float x2min = pred2.x;
    float x2max = pred2.x + pred2.width;
    float y2min = pred2.y;
    float y2max = pred2.y + pred2.height;
    float area2 = pred2.width * pred2.height;

    // get area for intersection box
    float x_inter_min = std::max(x1min, x2min);
    float x_inter_max = std::min(x1max, x2max);
    float y_inter_min = std::max(y1min, y2min);
    float y_inter_max = std::min(y1max, y2max);

    float width_inter = std::max(0.0f, x_inter_max - x_inter_min + 1);
    float height_inter = std::max(0.0f, y_inter_max - y_inter_min + 1);
    float area_inter = width_inter * height_inter;

    // return IoU
    return area_inter / (area1 + area2 - area_inter);
}

//calculate DIoU for 2 prediction boxes
float get_diou(t_prediction pred1, t_prediction pred2)
{
    // get bbox IoU
    float iou = get_iou(pred1, pred2);
    // coordinates for box 1
    float x1min = pred1.x;
    float x1max = pred1.x + pred1.width;
    float x1center = pred1.x + 0.5*pred1.width;
    float y1min = pred1.y;
    float y1max = pred1.y + pred1.height;
    float y1center = pred1.y + 0.5*pred1.height;

    // coordinates for box 2
    float x2min = pred2.x;
    float x2max = pred2.x + pred2.width;
    float x2center = pred2.x + 0.5*pred2.width;
    float y2min = pred2.y;
    float y2max = pred2.y + pred2.height;
    float y2center = pred2.y + 0.5*pred2.height;

    // box center distance
    float center_distance = pow(x1center - x2center, 2) + pow(y1center - y2center, 2);

    // get enclosed box
    float x_enclose_min = std::min(x1min, x2min);
    float x_enclose_max = std::max(x1max, x2max);
    float y_enclose_min = std::min(y1min, y2min);
    float y_enclose_max = std::max(y1max, y2max);
    // get enclosed diagonal distance
    float enclose_diagonal = pow(x_enclose_max - x_enclose_min, 2) + pow(y_enclose_max - y_enclose_min, 2);
    // calculate DIoU
    float diou = iou - 1.0 * center_distance / enclose_diagonal;
    return diou;
}


// less compare function for 2 prediction records (compare with confidence score),
// could be used for ascend order sort for prediction list
bool less_compare_pred(t_prediction lpred, t_prediction rpred)
{
    return (lpred.confidence < rpred.confidence);
}

// greater compare function for 2 prediction records (compare with confidence score),
// could be used for descend order sort for prediction list
bool greater_compare_pred(t_prediction lpred, t_prediction rpred)
{
    return (lpred.confidence > rpred.confidence);
}

// NMS operation for the prediction list
void nms_boxes(const std::vector<t_prediction> prediction_list, std::vector<t_prediction>& prediction_nms_list, int num_classes, float iou_threshold)
{
    MNN_PRINT("prediction_list size before NMS: %lu\n", prediction_list.size());
    //go through every class
    for (int i = 0; i < num_classes; i++) {

        //get prediction list for class i
        std::vector<t_prediction> class_pred_list;
        for (int j = 0; j < prediction_list.size(); j++) {
            if (prediction_list[j].class_index == i) {
                class_pred_list.emplace_back(prediction_list[j]);
            }
        }

        if(!class_pred_list.empty()) {
            std::vector<t_prediction> class_pick_list;
            // ascend sort the class prediction list
            std::sort(class_pred_list.begin(), class_pred_list.end(), less_compare_pred);

            while(class_pred_list.size() > 0) {
                // pick the max score prediction result
                t_prediction current_pred = class_pred_list.back();
                class_pick_list.emplace_back(current_pred);
                class_pred_list.pop_back();

                // loop the list to get IoU with max score prediction
                for(auto iter = class_pred_list.begin(); iter != class_pred_list.end();) {
                    // by default we use DIoU NMS
                    float iou = get_diou(current_pred, *iter);
                    // drop if IoU is larger than threshold
                    if(iou > iou_threshold) {
                        iter = class_pred_list.erase(iter);
                    } else {
                        iter++;
                    }
                }
            }

            // merge the picked predictions to final list
            prediction_nms_list.insert(prediction_nms_list.end(), class_pick_list.begin(), class_pick_list.end());
        }
    }

    return;
}


// select anchorset for corresponding featuremap layer
std::vector<std::pair<float, float>> get_anchorset(std::vector<std::pair<float, float>> anchors, const int stride)
{
    std::vector<std::pair<float, float>> anchorset;
    int anchor_num = anchors.size();

    // YOLOv3 model has 9 anchors and 3 feature layers
    if (anchor_num == 9) {
        if (stride == 32) {
            anchorset.emplace_back(anchors[6]);
            anchorset.emplace_back(anchors[7]);
            anchorset.emplace_back(anchors[8]);
        }
        else if (stride == 16) {
            anchorset.emplace_back(anchors[3]);
            anchorset.emplace_back(anchors[4]);
            anchorset.emplace_back(anchors[5]);
        }
        else if (stride == 8) {
            anchorset.emplace_back(anchors[0]);
            anchorset.emplace_back(anchors[1]);
            anchorset.emplace_back(anchors[2]);
        }
        else {
            MNN_PRINT("invalid feature map stride for anchorset!\n");
            exit(-1);
        }
    }
    // Tiny YOLOv3 model has 6 anchors and 2 feature layers
    else if (anchor_num == 6) {
        if (stride == 32) {
            anchorset.emplace_back(anchors[3]);
            anchorset.emplace_back(anchors[4]);
            anchorset.emplace_back(anchors[5]);
        }
        else if (stride == 16) {
            anchorset.emplace_back(anchors[0]);
            anchorset.emplace_back(anchors[1]);
            anchorset.emplace_back(anchors[2]);
        }
        else {
            MNN_PRINT("invalid anchorset index!\n");
            exit(-1);
        }
    }
    // YOLOv2 model has 5 anchors and 1 feature layers
    else if (anchor_num == 5) {
        anchorset = anchors;
    }
    else {
        MNN_PRINT("invalid anchor numbers!\n");
        exit(-1);
    }

    return anchorset;
}


// generate scale_x_y coefficient for "Eliminate grid sensitivity" feature,
// according to CLI option
float get_scale_x_y(const bool elim_grid_sense, std::vector<std::pair<float, float>> anchors, const int stride)
{
    int anchor_num = anchors.size();
    // by default return an invalid scale_x_y value
    float scale_x_y = -1;

    if(elim_grid_sense) {
        // YOLOv3 model has 9 anchors and 3 feature layers
        if (anchor_num == 9) {
            if (stride == 32) {
                scale_x_y = 1.05;
            }
            else if (stride == 16) {
                scale_x_y = 1.1;
            }
            else if (stride == 8) {
                scale_x_y = 1.1;
            }
            else {
                MNN_PRINT("invalid feature map stride!\n");
                exit(-1);
            }
        }
        // Tiny YOLOv3 model has 6 anchors and 2 feature layers
        else if (anchor_num == 6) {
            if (stride == 32) {
                scale_x_y = 1.05;
            }
            else if (stride == 16) {
                scale_x_y = 1.05;
            }
            else {
                MNN_PRINT("invalid anchorset index!\n");
                exit(-1);
            }
        }
        // YOLOv2 model has 5 anchors and 1 feature layers
        else if (anchor_num == 5) {
            scale_x_y = 1.05;
        }
        else {
            MNN_PRINT("invalid anchor numbers!\n");
            exit(-1);
        }
    }
    return scale_x_y;
}


void parse_anchors(std::string line, std::vector<std::pair<float, float>>& anchors)
{
    // parse anchor definition txt file
    // which should be like follow:
    //
    // yolo3_anchors:
    // 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
    //
    // tiny_yolo3_anchors:
    // 10,14,  23,27,  37,58,  81,82,  135,169,  344,319
    //
    // yolo2_anchors:
    // 18.32736, 21.67632, 59.98272, 66.00096, 106.82976, 175.17888, 252.25024, 112.88896, 312.65664, 293.38496
    //
    // yolo2-voc_anchors.txt:
    // 42.3072, 55.4064, 102.168, 128.30208, 161.78784, 259.16544, 303.07584, 154.89696, 359.5648, 320.2272
    //
    // yolo2-tiny_anchors.txt:
    // 18.32736, 21.67632, 59.98272, 66.00096, 106.82976, 175.17888, 252.25024, 112.88896, 312.65664, 293.38496
    //
    // yolo2-tiny-voc_anchors.txt
    // 34.56, 38.08, 109.44, 141.12, 212.16, 364.16, 301.44, 163.52, 531.84, 336.64
    size_t curr = 0, next = 0;

    while(next != std::string::npos) {
        //get 1st number
        next = line.find(",", curr);
        std::string num1 = line.substr(curr, next-curr);
        //get 2nd number
        curr = next + 1;
        next = line.find(",", curr);
        std::string num2 = line.substr(curr, next-curr);
        //form up anchor
        anchors.emplace_back(std::make_pair(atof(num1.c_str()), atof(num2.c_str())));
        //get start of next anchor
        curr = next + 1;
    }

    return;
}


void adjust_boxes(std::vector<t_prediction> &prediction_list, int image_width, int image_height, int input_width, int input_height)
{
    // Rescale the YOLO prediction (letterboxed) back to original image
    float ratio = std::min(float(input_width)/float(image_width), float(input_height)/float(image_height));

    float new_width = image_width * ratio;
    float new_height = image_height * ratio;

    float x_offset = (input_width - new_width) / 2.0 / input_width;
    float y_offset = (input_height - new_height) / 2.0 / input_height;
    float scale_x = input_width / new_width;
    float scale_y = input_height / new_height;

    for(auto &prediction : prediction_list) {
        prediction.x = (prediction.x - x_offset) * scale_x;
        prediction.y = (prediction.y - y_offset) * scale_y;
        prediction.width = prediction.width * scale_x;
        prediction.height = prediction.height * scale_y;

        // Convert centoids to top left coordinates
        prediction.x = prediction.x - (prediction.width / 2);
        prediction.y = prediction.y - (prediction.height / 2);

        // Scale boxes back to original image shape.
        prediction.x = prediction.x * image_width;
        prediction.y = prediction.y * image_height;
        prediction.width = prediction.width * image_width;
        prediction.height = prediction.height * image_height;
    }

    return;
}


//Resize image with unchanged aspect ratio using padding
uint8_t* letterbox_resize(uint8_t* inputImage, int image_width, int image_height, int image_channel, int input_width, int input_height, int input_channel)
{
    // assume the data channel match
    MNN_ASSERT(image_channel == input_channel);

    float scale = std::min(float(input_width)/float(image_width), float(input_height)/float(image_height));
    int padding_width = int(image_width * scale);
    int padding_height = int(image_height * scale);

    int x_offset = int((input_width - padding_width) / 2);
    int y_offset = int((input_height - padding_height) / 2);

    uint8_t* padding_image = (uint8_t*)malloc(padding_height * padding_width * image_channel * sizeof(uint8_t));
    if (padding_image == nullptr) {
        MNN_PRINT("Can't alloc memory\n");
        exit(-1);
    }

    stbir_resize_uint8(inputImage, image_width, image_height, 0,
                     padding_image, padding_width, padding_height, 0, image_channel);

    uint8_t* input_image = (uint8_t*)malloc(input_height * input_width * input_channel * sizeof(uint8_t));
    // here we need to fulfill the input image to 128 first, to align with the training preprocess
    memset((void*)input_image, 128, input_height * input_width * input_channel * sizeof(uint8_t));

    // paste input image into letterbox image
    for (int h = 0; h < padding_height; h++) {
        for (int w = 0; w < padding_width; w++) {
            for (int c = 0; c < image_channel; c++) {
                input_image[(h+y_offset)*input_width*input_channel + (w+x_offset)*input_channel + c] = padding_image[h*padding_width*image_channel + w*image_channel + c];
            }
        }
    }

    free(padding_image);
    return input_image;
}


template <typename T>
void fill_data(T* out, uint8_t* in, int input_width, int input_height,
            int input_channels, Settings* s) {
  auto output_number_of_pixels = input_height * input_width * input_channels;

  for (int i = 0; i < output_number_of_pixels; i++) {
    if (s->input_floating)
      out[i] = (in[i] - s->input_mean) / s->input_std;
    else
      out[i] = (uint8_t)in[i];
  }

  return;
}


void RunInference(Settings* s) {
    // record run time for every stage
    struct timeval start_time, stop_time;

    sigmoid_fast_init();

    // create model & session
    std::shared_ptr<Interpreter> net(Interpreter::createFromFile(s->model_name.c_str()));
    ScheduleConfig config;
    config.type  = MNN_FORWARD_AUTO; //MNN_FORWARD_CPU, MNN_FORWARD_OPENCL
    config.backupType = MNN_FORWARD_CPU;
    config.numThread = s->number_of_threads;

    BackendConfig bnconfig;
    bnconfig.memory = BackendConfig::Memory_Normal; //Memory_High, Memory_Low
    bnconfig.power = BackendConfig::Power_Normal; //Power_High, Power_Low
    bnconfig.precision = BackendConfig::Precision_Normal; //Precision_High, Precision_Low
    config.backendConfig = &bnconfig;

    auto session = net->createSession(config);

    // get input tensor info
    // assume only 1 input tensor (image_input)
    auto inputs = net->getSessionInputAll(session);
    MNN_ASSERT(inputs.size() == 1);
    auto image_input = inputs.begin()->second;
    int input_width = image_input->width();
    int input_height = image_input->height();
    int input_channel = image_input->channel();
    auto input_dim_type = image_input->getDimensionType();

    std::vector<std::string> dim_type_string = {"TENSORFLOW", "CAFFE", "CAFFE_C4"};

    MNN_PRINT("image_input: name:%s, width:%d, height:%d, channel:%d, dim_type:%s\n", inputs.begin()->first.c_str(), input_width, input_height, input_channel, dim_type_string[input_dim_type].c_str());

    // assume input tensor is Tensorflow format, NHWC
    // to align with the input image format
    //MNN_ASSERT(input_dim_type == Tensor::TENSORFLOW);

    auto shape = image_input->shape();
    shape[0] = 1;
    net->resizeTensor(image_input, shape);
    net->resizeSession(session);

    // since we don't need to create other sessions any more,
    // just release model data to save memory
    net->releaseModel();

    // get output tensor info (e.g. for YOLOv3 arch):
    //image_input: 1 x 416 x 416 x 3
    //"conv2d_3/Conv2D": 1 x 13 x 13 x 3 x (num_classes + 5)
    //"conv2d_8/Conv2D": 1 x 26 x 26 x 3 x (num_classes + 5)
    //"conv2d_13/Conv2D": 1 x 52 x 52 x 3 x (num_classes + 5)
    auto outputs = net->getSessionOutputAll(session);
    int num_layers = outputs.size();

    // get classes labels
    std::vector<std::string> classes;
    std::ifstream classesOs(s->classes_file_name.c_str());
    std::string line;
    while (std::getline(classesOs, line)) {
        classes.emplace_back(line);
    }
    int num_classes = classes.size();
    MNN_PRINT("num_classes: %d\n", num_classes);

    // get anchor value
    std::vector<std::pair<float, float>> anchors;
    std::ifstream anchorsOs(s->anchors_file_name.c_str());
    while (std::getline(anchorsOs, line)) {
        parse_anchors(line, anchors);
    }

    // For YOLOv3 model, we should have 9 anchors and 3 feature layers
    // For Tiny YOLOv3 model, we should have 6 anchors and 2 feature layers
    // For YOLOv2 model, we should have 5 anchors and 1 feature layers
    if(num_layers > 1)
        MNN_ASSERT(anchors.size() / num_layers == 3);

    // load input image
    auto inputPath = s->input_img_name.c_str();
    int image_width, image_height, image_channel;
    uint8_t* inputImage = (uint8_t*)stbi_load(inputPath, &image_width, &image_height, &image_channel, input_channel);
    if (nullptr == inputImage) {
        MNN_ERROR("Can't open %s\n", inputPath);
        return;
    }

    // do letterbox resize to input image
    uint8_t* letterboxImage = letterbox_resize(inputImage, image_width, image_height, image_channel, input_width, input_height, input_channel);

    // free input image
    stbi_image_free(inputImage);
    inputImage = nullptr;

    MNN_PRINT("origin image size: width:%d, height:%d, channel:%d\n", image_width, image_height, image_channel);

    // assume input tensor type is float
    MNN_ASSERT(image_input->getType().code == halide_type_float);
    s->input_floating = true;

    // create a host tensor for input data
    auto dataTensor = new Tensor(image_input, Tensor::TENSORFLOW);
    fill_data<float>(dataTensor->host<float>(), letterboxImage,
                input_width, input_height, input_channel, s);

    // run warm up session
    if (s->loop_count > 1)
        for (int i = 0; i < s->number_of_warmup_runs; i++) {
            image_input->copyFromHostTensor(dataTensor);
            if (net->runSession(session) != NO_ERROR) {
                MNN_PRINT("Failed to invoke MNN!\n");
            }
        }

    // run model sessions to get output
    gettimeofday(&start_time, nullptr);
    for (int i = 0; i < s->loop_count; i++) {
        image_input->copyFromHostTensor(dataTensor);
        if (net->runSession(session) != NO_ERROR) {
            MNN_PRINT("Failed to invoke MNN!\n");
        }
    }
    gettimeofday(&stop_time, nullptr);
    MNN_PRINT("model invoke average time: %lf ms\n", (get_us(stop_time) - get_us(start_time)) / (1000 * s->loop_count));


    // Copy output tensors to host, for further postprocess
    std::vector<std::shared_ptr<Tensor>> featureTensors;
    for(auto output : outputs) {
        auto output_tensor = output.second;
        int output_width = output_tensor->width();
        int output_height = output_tensor->height();
        int output_channel = output_tensor->channel();
        auto output_dim_type = output_tensor->getDimensionType();
        MNN_PRINT("output tensor: name:%s, width:%d, height:%d, channel:%d, dim_type:%s\n", output.first.c_str(), output_width, output_height, output_channel, dim_type_string[output_dim_type].c_str());

        if (output_tensor->getType().code != halide_type_float) {
            output_dim_type = Tensor::TENSORFLOW;
        }
        std::shared_ptr<Tensor> output_user(new Tensor(output_tensor, output_dim_type));
        output_tensor->copyToHostTensor(output_user.get());
        featureTensors.emplace_back(output_user);
    }

    // Do yolo_postprocess to parse out valid predictions
    std::vector<t_prediction> prediction_list;
    float conf_threshold = s->conf_thrd;
    float iou_threshold = 0.4;

    gettimeofday(&start_time, nullptr);

    for (int i = 0; i < num_layers; ++i) {
        Tensor* feature_map = featureTensors[i].get();

        // stride could confirm the feature map level:
        // image_input: 1 x 416 x 416 x 3
        // stride 32: 1 x 13 x 13 x 3 x (num_classes + 5)
        // stride 16: 1 x 26 x 26 x 3 x (num_classes + 5)
        // stride 8: 1 x 52 x 52 x 3 x (num_classes + 5)
        int stride = input_width / feature_map->width();

        std::vector<std::pair<float, float>> anchorset = get_anchorset(anchors, stride);

        float scale_x_y = get_scale_x_y(s->elim_grid_sense, anchors, stride);

        // Now we only support float32 type output tensor
        MNN_ASSERT(featureTensors[i]->getType().code == halide_type_float);
        MNN_ASSERT(featureTensors[i]->getType().bits == 32);
        //yolo_postprocess(featureTensors[i].get(), input_width, input_height, num_classes, anchorset, prediction_list, conf_threshold, scale_x_y);
        yolo_postprocess_fast(featureTensors[i].get(), input_width, input_height, num_classes, anchorset, prediction_list, conf_threshold, scale_x_y);
    }

    gettimeofday(&stop_time, nullptr);
    MNN_PRINT("yolo_postprocess time: %lf ms\n", (get_us(stop_time) - get_us(start_time)) / 1000);

    // Rescale the prediction back to original image
    adjust_boxes(prediction_list, image_width, image_height, input_width, input_height);

    // Do NMS for predictions
    std::vector<t_prediction> prediction_nms_list;
    gettimeofday(&start_time, nullptr);
    nms_boxes(prediction_list, prediction_nms_list, num_classes, iou_threshold);
    gettimeofday(&stop_time, nullptr);
    MNN_PRINT("NMS time: %lf ms\n", (get_us(stop_time) - get_us(start_time)) / 1000);

    // descend sort the final prediction result
    std::sort(prediction_nms_list.begin(), prediction_nms_list.end(), greater_compare_pred);

    // Open result txt file, in append mode
    std::ofstream resultOs (s->result_file_name.c_str(), std::ios::out | std::ios::app);
    // Get real path for input image
    char real_path_buff[PATH_MAX];
    if(realpath(inputPath, real_path_buff)) {
        resultOs << real_path_buff;
    } else {
        MNN_PRINT("fail to get image real path!\n");
        exit(-1);
    }

    // Show detection result
    MNN_PRINT("Detection result:\n");
    for(auto prediction_nms : prediction_nms_list) {
        // change box to (xmin,ymin,xmax,ymax) format
        int x_min = std::max(0, int(prediction_nms.x));
        int y_min = std::max(0, int(prediction_nms.y));
        int x_max = std::min(image_width, int(prediction_nms.x + prediction_nms.width));
        int y_max = std::min(image_height, int(prediction_nms.y + prediction_nms.height));

        MNN_PRINT("%s %f (%d, %d) (%d, %d)\n", classes[prediction_nms.class_index].c_str(), prediction_nms.confidence, x_min, y_min, x_max, y_max);
        // save detection result to file
        resultOs << " "
                 << x_min << ","
                 << y_min << ","
                 << x_max << ","
                 << y_max << ","
                 << prediction_nms.class_index << ","
                 << prediction_nms.confidence;
    }
    resultOs << "\n";
    resultOs.close();

    // Release buffer memory
    if (letterboxImage) {
        free(letterboxImage);
        letterboxImage = nullptr;
    }

    delete dataTensor;

    // Release session and model
    net->releaseSession(session);
    //net->releaseModel();
    return;
}


int main(int argc, char** argv) {
  Settings s;

  int c;
  while (1) {
    static struct option long_options[] = {
        {"mnn_model", required_argument, nullptr, 'm'},
        {"image", required_argument, nullptr, 'i'},
        {"classes", required_argument, nullptr, 'l'},
        {"anchors", required_argument, nullptr, 'a'},
        {"conf_thrd", required_argument, nullptr, 'n'},
        {"input_mean", required_argument, nullptr, 'b'},
        {"input_std", required_argument, nullptr, 's'},
        {"elim_grid_sense", required_argument, nullptr, 'e'},
        {"threads", required_argument, nullptr, 't'},
        {"count", required_argument, nullptr, 'c'},
        {"warmup_runs", required_argument, nullptr, 'w'},
        {"result", required_argument, nullptr, 'r'},
        //{"verbose", required_argument, nullptr, 'v'},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0}};

    /* getopt_long stores the option index here. */
    int option_index = 0;

    c = getopt_long(argc, argv,
                    "a:b:c:e:hi:l:m:n:r:s:t:w:", long_options,
                    &option_index);

    /* Detect the end of the options. */
    if (c == -1) break;

    switch (c) {
      case 'a':
        s.anchors_file_name = optarg;
        break;
      case 'b':
        s.input_mean = strtod(optarg, nullptr);
        break;
      case 'c':
        s.loop_count =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'e':
        s.elim_grid_sense =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'i':
        s.input_img_name = optarg;
        break;
      case 'l':
        s.classes_file_name = optarg;
        break;
      case 'm':
        s.model_name = optarg;
        break;
      case 'n':
        s.conf_thrd = strtod(optarg, nullptr);
        break;
      case 's':
        s.input_std = strtod(optarg, nullptr);
        break;
      case 't':
        s.number_of_threads = strtol(  // NOLINT(runtime/deprecated_fn)
            optarg, nullptr, 10);
        break;
      //case 'v':
        //s.verbose =
            //strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        //break;
      case 'w':
        s.number_of_warmup_runs =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'r':
        s.result_file_name = optarg;
        break;
      case 'h':
      case '?':
      default:
        /* getopt_long already printed an error message. */
        display_usage();
        exit(-1);
    }
  }
  RunInference(&s);
  return 0;
}

