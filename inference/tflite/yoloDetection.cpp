//
//  yoloDetection.cpp
//  Tensorflow-lite
//
//  Created by Xiaobin Zhang on 2019/09/20.
//

#include <fcntl.h>
#include <math.h>
#include <getopt.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>
#include <assert.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>
#include <numeric>
#include <algorithm>

#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/string_util.h"

#include "yoloDetection.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

#define LOG(x) std::cerr

namespace yoloDetection {

// definition of a bbox prediction record
typedef struct prediction {
    float x;
    float y;
    float width;
    float height;
    float confidence;
    int class_index;
}t_prediction;

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


// YOLO postprocess for each prediction feature map
void yolo_postprocess(const TfLiteTensor* feature_map, const int input_width, const int input_height,
                      const int num_classes, const std::vector<std::pair<float, float>> anchors,
                      std::vector<t_prediction> &prediction_list, float conf_threshold)
{
    // 1. do following transform to get the output bbox,
    //    which is aligned with YOLOv3/YOLOv2 paper:
    //
    //    bbox_x = sigmoid(pred_x) + grid_w
    //    bbox_y = sigmoid(pred_y) + grid_h
    //    bbox_w = exp(pred_w) * anchor_w / stride
    //    bbox_h = exp(pred_h) * anchor_h / stride
    //    bbox_obj = sigmoid(pred_obj)
    //
    // 2. convert the grid scale coordinate back to
    //    input image shape, with stride:
    //
    //    bbox_x = bbox_x * stride;
    //    bbox_y = bbox_y * stride;
    //    bbox_w = bbox_w * stride;
    //    bbox_h = bbox_h * stride;
    //
    // 3. convert centoids to top left coordinates
    //
    //    bbox_x = bbox_x - (bbox_w / 2);
    //    bbox_y = bbox_y - (bbox_h / 2);
    //
    // 4. get bbox confidence (class_score * objectness)
    //    and filter with threshold
    //
    //    bbox_conf[:] = sigmoid/softmax(bbox_class_score[:]) * bbox_obj
    //    bbox_max_conf = max(bbox_conf[:])
    //    bbox_max_index = argmax(bbox_conf[:])
    //
    // 5. filter bbox_max_conf with threshold
    //
    //    if(bbox_max_conf > conf_threshold)
    //        enqueue the bbox info

    const float* data = reinterpret_cast<float*>(feature_map->data.raw);

    TfLiteIntArray* output_dims = feature_map->dims;

    int batch = output_dims->data[0];
    int height = output_dims->data[1];
    int width = output_dims->data[2];
    int channel = output_dims->data[3];

    int stride = input_width / width;
    auto unit = sizeof(float);
    int anchor_num_per_layer = anchors.size();

    // TF/TFLite tensor format: NHWC
    auto bytesPerRow   = channel * unit;
    auto bytesPerImage = width * bytesPerRow;
    auto bytesPerBatch = height * bytesPerImage;

    for (int b = 0; b < batch; b++) {
        auto bytes = data + b * bytesPerBatch / unit;
        LOG(INFO) << "batch " << b << "\n";

        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                for (int anc = 0; anc < anchor_num_per_layer; anc++) {
                    //get bbox prediction data for each anchor, each feature point
                    int bbox_x_offset = h * width * channel + w * channel + anc * (num_classes + 5);
                    int bbox_y_offset = h * width * channel + w * channel + anc * (num_classes + 5) + 1;
                    int bbox_w_offset = h * width * channel + w * channel + anc * (num_classes + 5) + 2;
                    int bbox_h_offset = h * width * channel + w * channel + anc * (num_classes + 5) + 3;
                    int bbox_obj_offset = h * width * channel + w * channel + anc * (num_classes + 5) + 4;
                    int bbox_scores_offset = h * width * channel + w * channel + anc * (num_classes + 5) + 5;
                    int bbox_scores_step = 1;

                    float bbox_x = sigmoid(bytes[bbox_x_offset]) + w;
                    float bbox_y = sigmoid(bytes[bbox_y_offset]) + h;
                    float bbox_w = exp(bytes[bbox_w_offset]) * anchors[anc].first / stride;
                    float bbox_h = exp(bytes[bbox_h_offset]) * anchors[anc].second / stride;
                    float bbox_obj = sigmoid(bytes[bbox_obj_offset]);

                    // Transfer anchor coordinates
                    bbox_x = bbox_x * stride;
                    bbox_y = bbox_y * stride;
                    bbox_w = bbox_w * stride;
                    bbox_h = bbox_h * stride;

                    // Convert centoids to top left coordinates
                    bbox_x = bbox_x - (bbox_w / 2);
                    bbox_y = bbox_y - (bbox_h / 2);

                    // Get softmax score for YOLOv2 prediction
                    std::vector<float> logits_bbox_score;
                    std::vector<float> bbox_score;
                    if(anchor_num_per_layer == 5) {
                        for (int i = 0; i < num_classes; i++) {
                            logits_bbox_score.emplace_back(bytes[bbox_scores_offset + i * bbox_scores_step]);
                        }
                        softmax(logits_bbox_score, bbox_score);
                    }

                    //get anchor output confidence (class_score * objectness) and filter with threshold
                    float max_conf = 0.0;
                    int max_index = -1;
                    for (int i = 0; i < num_classes; i++) {
                        float tmp_conf = 0.0;
                        if(anchor_num_per_layer == 5) {
                            // YOLOv2 use 5 anchors and softmax class scores
                            tmp_conf = bbox_score[i] * bbox_obj;
                        }
                        else {
                            tmp_conf = sigmoid(bytes[bbox_scores_offset + i * bbox_scores_step]) * bbox_obj;
                        }

                        if(tmp_conf > max_conf) {
                            max_conf = tmp_conf;
                            max_index = i;
                        }
                    }
                    if(max_conf >= conf_threshold) {
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


// select anchorset for corresponding featuremap layer
std::vector<std::pair<float, float>> get_anchorset(std::vector<std::pair<float, float>> anchors, const int feature_width, const int input_width)
{
    std::vector<std::pair<float, float>> anchorset;
    int anchor_num = anchors.size();

    // stride could confirm the feature map level:
    // image_input: 1 x 416 x 416 x 3
    // stride 32: 1 x 13 x 13 x 3 x (num_classes + 5)
    // stride 16: 1 x 26 x 26 x 3 x (num_classes + 5)
    // stride 8: 1 x 52 x 52 x 3 x (num_classes + 5)
    int stride = input_width / feature_width;

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
            LOG(ERROR) << "invalid feature map stride for anchorset!\n";
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
            LOG(ERROR) << "invalid anchorset index!\n";
            exit(-1);
        }
    }
    // YOLOv2 model has 5 anchors and 1 feature layers
    else if (anchor_num == 5) {
        anchorset = anchors;
    }
    else {
        LOG(ERROR) << "invalid anchor numbers!\n";
        exit(-1);
    }

    return anchorset;
}

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

//ascend order sort for prediction records
bool compare_conf(t_prediction lpred, t_prediction rpred)
{
    if (lpred.confidence < rpred.confidence)
        return true;
    else
        return false;
}


// NMS operation for the prediction list
void nms_boxes(const std::vector<t_prediction> prediction_list, std::vector<t_prediction>& prediction_nms_list, int num_classes, float iou_threshold)
{
    printf("prediction_list size before NMS: %lu\n", prediction_list.size());
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
            std::sort(class_pred_list.begin(), class_pred_list.end(), compare_conf);

            while(class_pred_list.size() > 0) {
                // pick the max score prediction result
                t_prediction current_pred = class_pred_list.back();
                class_pick_list.emplace_back(current_pred);
                class_pred_list.pop_back();

                // loop the list to get IoU with max score prediction
                for(auto iter = class_pred_list.begin(); iter != class_pred_list.end();) {
                    float iou = get_iou(current_pred, *iter);
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


void adjust_boxes(std::vector<t_prediction> &prediction_nms_list, int image_width, int image_height, int input_width, int input_height)
{
    // Rescale the final prediction (letterboxed) back to original image
    assert(input_width == input_height);

    int square_dim = std::max(image_width, image_height);
    float scale = float(square_dim) / float(input_width);
    int x_offset, y_offset;

    if ( image_width > image_height ) {
        x_offset = 0;
        y_offset = floor((image_width - image_height) / 2);
    }
    else {
        x_offset = floor((image_height - image_width) / 2);
        y_offset = 0;
    }


    for(auto &prediction_nms : prediction_nms_list) {
        prediction_nms.x = prediction_nms.x * scale - x_offset;
        prediction_nms.y = prediction_nms.y * scale - y_offset;
        prediction_nms.width = prediction_nms.width * scale;
        prediction_nms.height = prediction_nms.height * scale;
    }

    return;
}

//Resize image with unchanged aspect ratio using padding
uint8_t* letterbox_image(uint8_t* inputImage, int image_width, int image_height, int image_channel)
{
    // if input image is square, just return original
    if (image_width == image_height) {
        return inputImage;
    }

    int square_dim = std::max(image_width, image_height);
    int x_offset, y_offset;

    uint8_t* squareImage = (uint8_t*)malloc(square_dim * square_dim * image_channel * sizeof(uint8_t));

    if ( image_width > image_height ) {
        x_offset = 0;
        y_offset = floor((image_width - image_height) / 2);
    }
    else {
        x_offset = floor((image_height - image_width) / 2);
        y_offset = 0;
    }

    // paste input image into square image
    for (int h = 0; h < image_height; h++) {
        for (int w = 0; w < image_width; w++) {
            for (int c = 0; c < image_channel; c++) {
                squareImage[(h+y_offset)*square_dim*image_channel + (w+x_offset)*image_channel + c] = inputImage[h*image_width*image_channel + w*image_channel + c];
            }
        }
    }

    return squareImage;
}


template <class T>
void resize(T* out, uint8_t* in, int image_width, int image_height,
            int image_channels, int wanted_width, int wanted_height,
            int wanted_channels, Settings* s) {
  uint8_t* resized = (uint8_t*)malloc(wanted_height * wanted_width * wanted_channels * sizeof(uint8_t));
  if (resized == nullptr) {
      LOG(FATAL) << "Can't alloc memory" << "\n";
      exit(-1);
  }

  stbir_resize_uint8(in, image_width, image_height, 0,
                     resized, wanted_width, wanted_height, 0, wanted_channels);

  auto output_number_of_pixels = wanted_height * wanted_width * wanted_channels;

  for (int i = 0; i < output_number_of_pixels; i++) {
    if (s->input_floating)
      out[i] = (resized[i] - s->input_mean) / s->input_std;
    else
      out[i] = (uint8_t)resized[i];
  }

  free(resized);
  return;
}


void RunInference(Settings* s) {
  if (!s->model_name.c_str()) {
    LOG(ERROR) << "no model file name\n";
    exit(-1);
  }

  // load model
  std::unique_ptr<tflite::FlatBufferModel> model;
  std::unique_ptr<tflite::Interpreter> interpreter;
  model = tflite::FlatBufferModel::BuildFromFile(s->model_name.c_str());
  if (!model) {
    LOG(FATAL) << "\nFailed to mmap model " << s->model_name << "\n";
    exit(-1);
  }
  s->model = model.get();
  LOG(INFO) << "Loaded model " << s->model_name << "\n";
  model->error_reporter();
  LOG(INFO) << "resolved reporter\n";


  // prepare model interpreter
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  if (!interpreter) {
    LOG(FATAL) << "Failed to construct interpreter\n";
    exit(-1);
  }

  interpreter->SetAllowFp16PrecisionForFp32(s->allow_fp16);
  if (s->number_of_threads != -1) {
    interpreter->SetNumThreads(s->number_of_threads);
  }

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    LOG(FATAL) << "Failed to allocate tensors!";
  }


  // get classes labels
  std::vector<std::string> classes;
  std::ifstream classesOs(s->classes_file_name.c_str());
  std::string line;
  while (std::getline(classesOs, line)) {
      classes.emplace_back(line);
  }
  int num_classes = classes.size();
  LOG(INFO) << "num_classes: " << num_classes << "\n";


  // get anchor value
  std::vector<std::pair<float, float>> anchors;
  std::ifstream anchorsOs(s->anchors_file_name.c_str());
  while (std::getline(anchorsOs, line)) {
      parse_anchors(line, anchors);
  }


  // read input image
  int image_width = 224;
  int image_height = 224;
  int image_channel = 3;

  auto input_image = (uint8_t*)stbi_load(s->input_img_name.c_str(), &image_width, &image_height, &image_channel, 3);
  if (input_image == nullptr) {
      LOG(FATAL) << "Can't open" << s->input_img_name << "\n";
      exit(-1);
  }

  // pad input image to letterboxed for input resize
  uint8_t* letterboxImage = letterbox_image(input_image, image_width, image_height, image_channel);
  int square_dim = std::max(image_width, image_height);

  std::vector<uint8_t> in(letterboxImage, letterboxImage + square_dim * square_dim * image_channel * sizeof(uint8_t));

  // free input image
  stbi_image_free(input_image);
  if(letterboxImage != input_image) {
      free(letterboxImage);
  }
  input_image = nullptr;
  letterboxImage = nullptr;

  // assuming one input only
  int input = interpreter->inputs()[0];

  LOG(INFO) << "origin image size: width:" << image_width
            << ", height:" << image_height
            << ", channel:" << image_channel
            << "\n";

  // get input dimension from the input tensor metadata
  TfLiteIntArray* dims = interpreter->tensor(input)->dims;
  int input_batch = dims->data[0];
  int input_height = dims->data[1];
  int input_width = dims->data[2];
  int input_channels = dims->data[3];

  if (s->verbose) LOG(INFO) << "input tensor info: "
                            << "type " << interpreter->tensor(input)->type << ", "
                            << "batch " << input_batch << ", "
                            << "height " << input_height << ", "
                            << "width " << input_width << ", "
                            << "channels " << input_channels << "\n";
  // assume the model input is square
  assert(input_width == input_height);

  // resize image to model input shape
  switch (interpreter->tensor(input)->type) {
    case kTfLiteFloat32:
      s->input_floating = true;
      resize<float>(interpreter->typed_tensor<float>(input), in.data(),
                    square_dim, square_dim, image_channel, input_width,
                    input_height, input_channels, s);
      break;
    case kTfLiteUInt8:
      resize<uint8_t>(interpreter->typed_tensor<uint8_t>(input), in.data(),
                      square_dim, square_dim, image_channel, input_width,
                      input_height, input_channels, s);
      break;
    default:
      LOG(FATAL) << "cannot handle input type "
                 << interpreter->tensor(input)->type << " yet";
      exit(-1);
  }


  // run warm up session
  if (s->loop_count > 1)
    for (int i = 0; i < s->number_of_warmup_runs; i++) {
      if (interpreter->Invoke() != kTfLiteOk) {
        LOG(FATAL) << "Failed to invoke tflite!\n";
      }
    }

  // run model sessions to get output
  struct timeval start_time, stop_time;
  gettimeofday(&start_time, nullptr);
  for (int i = 0; i < s->loop_count; i++) {
    if (interpreter->Invoke() != kTfLiteOk) {
      LOG(FATAL) << "Failed to invoke tflite!\n";
    }
  }
  gettimeofday(&stop_time, nullptr);
  LOG(INFO) << "invoked average time:" << (get_us(stop_time) - get_us(start_time)) / (s->loop_count * 1000) << " ms \n";


  // Do yolo_postprocess to parse out valid predictions
  std::vector<t_prediction> prediction_list;
  float conf_threshold = 0.1;
  float iou_threshold = 0.4;
  const std::vector<int> outputs = interpreter->outputs();

  gettimeofday(&start_time, nullptr);
  for (int i = 0; i < outputs.size(); i++) {
      int output = interpreter->outputs()[i];
      TfLiteIntArray* output_dims = interpreter->tensor(output)->dims;

      int output_batch = output_dims->data[0];
      int output_height = output_dims->data[1];
      int output_width = output_dims->data[2];
      int output_channels = output_dims->data[3];

      if (s->verbose) LOG(INFO) << "output tensor info: "
                                << "name " << interpreter->tensor(output)->name << ", "
                                << "type " << interpreter->tensor(output)->type << ", "
                                << "batch " << output_batch << ", "
                                << "height " << output_height << ", "
                                << "width " << output_width << ", "
                                << "channels " << output_channels << "\n";
      TfLiteTensor* feature_map = interpreter->tensor(output);
      std::vector<std::pair<float, float>> anchorset = get_anchorset(anchors, feature_map->dims->data[2], input_width);

      // Now we only support float32 type output tensor
      assert(feature_map->type == kTfLiteFloat32);
      yolo_postprocess(feature_map, input_width, input_height, num_classes, anchorset, prediction_list, conf_threshold);
  }
  gettimeofday(&stop_time, nullptr);
  LOG(INFO) << "yolo_postprocess time: " << (get_us(stop_time) - get_us(start_time)) / 1000 << " ms\n";


  // Do NMS for predictions
  std::vector<t_prediction> prediction_nms_list;
  gettimeofday(&start_time, nullptr);
  nms_boxes(prediction_list, prediction_nms_list, num_classes, iou_threshold);
  gettimeofday(&stop_time, nullptr);
  LOG(INFO) << "NMS time: " << (get_us(stop_time) - get_us(start_time)) / 1000 << " ms\n";

  // Rescale the prediction back to original image
  adjust_boxes(prediction_nms_list, image_width, image_height, input_width, input_height);

  // Show detection result
  LOG(INFO) << "Detection result:\n";
  for(auto prediction_nms : prediction_nms_list) {
      LOG(INFO) << classes[prediction_nms.class_index] << " "
                << prediction_nms.confidence << " "
                << "(" << int(prediction_nms.x) << ", " << int(prediction_nms.y) << ")"
                << " (" << int(prediction_nms.x + prediction_nms.width) << ", " << int(prediction_nms.y + prediction_nms.height) << ")\n";
  }

  return;
}

void display_usage() {
  LOG(INFO)
      << "Usage: yoloDetection\n"
      << "--tflite_model, -m: model_name.tflite\n"
      << "--image, -i: image_name.jpg\n"
      << "--classes, -l: classes labels for the model\n"
      << "--anchors, -a: anchor values for the model\n"
      << "--input_mean, -b: input mean\n"
      << "--input_std, -s: input standard deviation\n"
      << "--allow_fp16, -f: [0|1], allow running fp32 models with fp16 or not\n"
      << "--threads, -t: number of threads\n"
      << "--count, -c: loop interpreter->Invoke() for certain times\n"
      << "--warmup_runs, -w: number of warmup runs\n"
      << "--verbose, -v: [0|1] print more information\n"
      << "\n";
}

int Main(int argc, char** argv) {
  Settings s;

  int c;
  while (1) {
    static struct option long_options[] = {
        {"tflite_model", required_argument, nullptr, 'm'},
        {"image", required_argument, nullptr, 'i'},
        {"classes", required_argument, nullptr, 'l'},
        {"anchors", required_argument, nullptr, 'a'},
        {"input_mean", required_argument, nullptr, 'b'},
        {"input_std", required_argument, nullptr, 's'},
        {"threads", required_argument, nullptr, 't'},
        {"allow_fp16", required_argument, nullptr, 'f'},
        {"count", required_argument, nullptr, 'c'},
        {"warmup_runs", required_argument, nullptr, 'w'},
        {"verbose", required_argument, nullptr, 'v'},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0}};

    /* getopt_long stores the option index here. */
    int option_index = 0;

    c = getopt_long(argc, argv,
                    "a:b:c:f:hi:l:m:s:t:v:w:", long_options,
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
      case 'f':
        s.allow_fp16 =
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
      case 's':
        s.input_std = strtod(optarg, nullptr);
        break;
      case 't':
        s.number_of_threads = strtol(  // NOLINT(runtime/deprecated_fn)
            optarg, nullptr, 10);
        break;
      case 'v':
        s.verbose =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'w':
        s.number_of_warmup_runs =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'h':
      case '?':
      default:
        /* getopt_long already printed an error message. */
        display_usage();
        exit(-1);
        exit(-1);
    }
  }
  RunInference(&s);
  return 0;
}

}  // namespace yoloDetection

int main(int argc, char** argv) {
  return yoloDetection::Main(argc, argv);
}
