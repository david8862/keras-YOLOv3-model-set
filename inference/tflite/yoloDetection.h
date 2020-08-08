//
//  yoloDetection.h
//  Tensorflow-lite
//
//  Created by Xiaobin Zhang on 2019/09/20.
//
//

#ifndef YOLO_DETECTION_YOLO_DETECTION_H_
#define YOLO_DETECTION_YOLO_DETECTION_H_

#include "tensorflow/lite/model.h"
#include "tensorflow/lite/string_type.h"

namespace yoloDetection {

struct Settings {
  bool verbose = false;
  bool accel = false;
  bool input_floating = false;
  bool allow_fp16 = false;
  int loop_count = 1;
  float conf_thrd = 0.1f;
  float input_mean = 0.0f;
  float input_std = 255.0f;
  bool elim_grid_sense = false;
  std::string model_name = "./model.tflite";
  tflite::FlatBufferModel* model;
  std::string input_img_name = "./dog.jpg";
  std::string classes_file_name = "./classes.txt";
  std::string anchors_file_name = "./yolo3_anchors.txt";
  std::string result_file_name = "./result.txt";
  std::string input_layer_type = "uint8_t";
  int number_of_threads = 4;
  int number_of_warmup_runs = 2;
};

}  // namespace yoloDetection

#endif  // YOLO_DETECTION_YOLO_DETECTION_H_
