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

#ifndef YOLOV3_DETECTION_RESIZE_HELPERS_H_
#define YOLOV3_DETECTION_RESIZE_HELPERS_H_

#include "resize_helpers_impl.h"
#include "yolov3Detection.h"

namespace yolov3Detection {

template <class T>
void resize(T* out, uint8_t* in, int image_height, int image_width,
            int image_channels, int wanted_height, int wanted_width,
            int wanted_channels, Settings* s);

// explicit instantiation
template void resize<uint8_t>(uint8_t*, unsigned char*, int, int, int, int, int,
                              int, Settings*);
template void resize<float>(float*, unsigned char*, int, int, int, int, int,
                            int, Settings*);

}  // namespace yolov3Detection

#endif  // YOLOV3_DETECTION_RESIZE_HELPERS_H_
