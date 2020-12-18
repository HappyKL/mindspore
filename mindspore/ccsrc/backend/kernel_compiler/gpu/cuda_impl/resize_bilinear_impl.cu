/**
 * Copyright 2019 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <algorithm>
#include "backend/kernel_compiler/gpu/cuda_impl/resize_bilinear_impl.cuh"

template <typename T>
__global__ void ResizeBilinear(const int size, const T *input, const int s1, const int s2, const int s3,
                                      const int s4, T *output, const int d1, const int d2, const int d3, const int d4,
                                      bool align_corners, float h_scale, float w_scale) {
  // initialization
  // HalfPixelCenters false
  int input_pos1;
  int input_pos2;
  int input_pos3;
  int input_pos4;
  int pos_array[RESIZEBILINEAR_DIMENSION];
//  int in_height = s3;
//  int in_width = s4;
  // for example 4-D: pos = pos_array[0] * output_shape[1] * output_shape[2] * output_shape[3] +
  //                        pos_array[1] * output_shape[2] * output_shape[3] +
  //                        pos_array[2] * output_shape[3] +
  //                        pos_array[3]
//  int out_h;
//  int out_w;

  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    pos_array[0] = pos / (d2 * d3 * d4) % d1;
    pos_array[1] = pos / (d3 * d4) % d2;
    pos_array[2] = pos / (d4) % d3;
    pos_array[3] = pos % d4;
//    out_h = pos_array[2];
//    out_w = pos_array[3];


    int w2 = pos % d4; // 0:width2-1
    int h2 = pos / d4; // 0:height2-1


    T hlr;
    if (align_corners) {
        hlr =  static_cast<T>h_scale * static_cast<T>h2;
    } else {
        T src_idx = static_cast<T>h_scale * (static_cast<T>h2 + static_cast<T>(0.5)) - static_cast<T>(0.5);
        // See Note[Follow Opencv resize logic]
        hlr = (src_idx < static_cast<T>(0))
        ? static_cast<T>(0)
        : src_idx;
    }
    const int h1 = hlr;
    const int hlp = (h1 < s3 - 1) ? 1:0;
    const T h1lambda = hlr - static_cast<T>h1;
    const T h0lambda = static_cast<T>(1) - h1lambda;


    T wlr;
    if (align_corners) {
        wlr =  static_cast<T>w_scale * static_cast<T>w2;
    } else {
        T src_idx = static_cast<T>w_scale * (static_cast<T>w2 + static_cast<T>(0.5)) - static_cast<T>(0.5);
        // See Note[Follow Opencv resize logic]
        wlr = (src_idx < static_cast<T>(0))
        ? static_cast<T>(0)
        : src_idx;
    }
    const int w1 = wlr;
    const int wlp = (w1 < s4 -1 )? 1:0;
    const T w1lambda = wlr -static_cast<T> w1;
    const T w0lambda = static_cast<T>(1) - w1lambda;

    input_pos1 = pos_array[0] * s2 * s3 * s4 + pos_array[1] * s3 * s4 + h1 * s4 + w1;
    input_pos2 = pos_array[0] * s2 * s3 * s4 + pos_array[1] * s3 * s4 + h1 * s4 + w1 + wlp;
    input_pos3 = pos_array[0] * s2 * s3 * s4 + pos_array[1] * s3 * s4 + (h1 + hlp) * s4 + w1;
    input_pos4 = pos_array[0] * s2 * s3 * s4 + pos_array[1] * s3 * s4 + (h1 + hlp) * s4 + w1 + wlp;

    const T val = h0lambda *
                (w0lambda * input[input_pos1] +
                 w1lambda * input[input_pos2]) +
            h1lambda *
                (w0lambda * input[input_pos3] +
                 w1lambda * input[input_pos4]);

//    const int in_y =
//      min((align_corners) ? static_cast<int>(roundf(out_h * h_scale)) : static_cast<int>(floorf(out_h * h_scale)),
//          in_height - 1);
//    const int in_x =
//      min((align_corners) ? static_cast<int>(roundf(out_w * w_scale)) : static_cast<int>(floorf(out_w * w_scale)),
//          in_width - 1);
//    // pos_array[0] N, pos_array[1] C, in_y H, in_x W
//    input_pos = pos_array[0] * s2 * s3 * s4 + pos_array[1] * s3 * s4 + in_y * s4 + in_x;
//    output[pos] = input[input_pos];
      output[pos] = val;
  }
  return;
}

template <typename T>
void CalResizeBilinear(const int size, const T *input, const int s1, const int s2, const int s3, const int s4,
                              T *output, const int d1, const int d2, const int d3, const int d4, bool align_corners,
                              float h_scale, float w_scale, cudaStream_t cuda_stream) {
  ResizeBilinear<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, input, s1, s2, s3, s4, output, d1, d2,
                                                                           d3, d4, align_corners, h_scale, w_scale);
  return;
}

template void CalResizeBilinear<float>(const int size, const float *input, const int s1, const int s2,
                                              const int s3, const int s4, float *output, const int d1, const int d2,
                                              const int d3, const int d4, bool align_corners, float h_scale,
                                              float w_scale, cudaStream_t cuda_stream);
template void CalResizeBilinear<half>(const int size, const half *input, const int s1, const int s2,
                                             const int s3, const int s4, half *output, const int d1, const int d2,
                                             const int d3, const int d4, bool align_corners, float h_scale,
                                             float w_scale, cudaStream_t cuda_stream);
template void CalResizeBilinear<int>(const int size, const int *input, const int s1, const int s2, const int s3,
                                            const int s4, int *output, const int d1, const int d2, const int d3,
                                            const int d4, bool align_corners, float h_scale, float w_scale,
                                            cudaStream_t cuda_stream);
