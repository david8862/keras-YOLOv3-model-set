/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef YOLOV3_DETECTION_YOLOV3_DETECTION_H_
#define YOLOV3_DETECTION_YOLOV3_DETECTION_H_

#include "tensorflow/lite/model.h"
#include "tensorflow/lite/string_type.h"

namespace yolov3Detection {

struct Settings {
  bool verbose = false;
  bool accel = false;
  bool input_floating = false;
  bool allow_fp16 = false;
  int loop_count = 1;
  float input_mean = 0.0f;
  float input_std = 255.0f;
  std::string model_name = "./model.tflite";
  tflite::FlatBufferModel* model;
  std::string input_img_name = "./dog.jpg";
  std::string classes_file_name = "./classes.txt";
  std::string anchors_file_name = "./yolo_anchors.txt";
  std::string input_layer_type = "uint8_t";
  int number_of_threads = 4;
  int number_of_warmup_runs = 2;
};

}  // namespace yolov3Detection

#endif  // YOLOV3_DETECTION_YOLOV3_DETECTION_H_
