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
#include "backend/kernel_compiler/gpu/cuda_impl/util.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/resize_bilinear_grad_impl.cuh"

template <typename T>
__global__ void InitZero(T *output, const int output_size) {
  for (size_t pos = threadIdx.x + blockIdx.x * blockDim.x; pos < (output_size); pos += gridDim.x * blockDim.x) {
    output[pos] = static_cast<T>(0);
  }
}

template <typename T>
__global__ void ResizeBilinearGrad(const int input_size, const T *input, const int s1, const int s2,
                                          const int s3, const int s4, T *output, const int d1, const int d2,
                                          const int d3, const int d4, bool align_corners, float h_scale,
                                          float w_scale) {
  // initialization
  // HalfPixelCenters false
  int output_pos1;
  int output_pos2;
  int output_pos3;
  int output_pos4;
  int pos_array[RESIZEBILINEAR_DIMENSION];
//  int out_height = d3;
//  int out_width = d4;
  // for example 4-D: pos = pos_array[0] * output_shape[1] * output_shape[2] * output_shape[3] +
  //                        pos_array[1] * output_shape[2] * output_shape[3] +
  //                        pos_array[2] * output_shape[3] +
  //                        pos_array[3]
  int in_h;
  int in_w;

  for (size_t pos = threadIdx.x + blockIdx.x * blockDim.x; pos < (input_size); pos += gridDim.x * blockDim.x) {
    pos_array[0] = pos / (s2 * s3 * s4) % s1;
    pos_array[1] = pos / (s3 * s4) % s2;
    pos_array[2] = pos / (s4) % s3;
    pos_array[3] = pos % s4;
    in_h = pos_array[2];
    in_w = pos_array[3];

     T hlr;
    if (align_corners) {
        hlr =  static_cast<T>(h_scale) * static_cast<T>(in_h);
    } else {
        T src_idx = static_cast<T>(h_scale) * (static_cast<T>(in_h) + static_cast<T>(0.5)) - static_cast<T>(0.5);
        // See Note[Follow Opencv resize logic]
        hlr = (src_idx < static_cast<T>(0))
        ? static_cast<T>(0)
        : src_idx;
    }
    const int h1 = hlr;
    const int hlp = (h1 < d3 - 1) ? 1:0;
    const T h1lambda = hlr - static_cast<T>(in_h);
    const T h0lambda = static_cast<T>(1) - h1lambda;

     T wlr;
    if (align_corners) {
        wlr =  static_cast<T>(w_scale) * static_cast<T>(in_w);
    } else {
        T src_idx = static_cast<T>(w_scale) * (static_cast<T>(in_w) + static_cast<T>(0.5)) - static_cast<T>(0.5);
        // See Note[Follow Opencv resize logic]
        wlr = (src_idx < static_cast<T>(0))
        ? static_cast<T>(0)
        : src_idx;
    }
    const int w1 = wlr;
    const int wlp = (w1 < d4 -1 )? 1:0;
    const T w1lambda = wlr -static_cast<T>(in_w);
    const T w0lambda = static_cast<T>(1) - w1lambda;

    output_pos1 = pos_array[0] * d2 * d3 * d4 + pos_array[1] * d3 * d4 + h1 * d4 + w1;
    output_pos2 = pos_array[0] * d2 * d3 * d4 + pos_array[1] * d3 * d4 + h1 * d4 + w1 + wlp;
    output_pos3 = pos_array[0] * d2 * d3 * d4 + pos_array[1] * d3 * d4 + (h1 + hlp) * d4 + w1;
    output_pos4 = pos_array[0] * d2 * d3 * d4 + pos_array[1] * d3 * d4 + (h1 + hlp) * d4 + w1 + wlp;

    MsAtomicAdd(&output[output_pos1], static_cast<T>(h0lambda * w0lambda * input[pos]));
    MsAtomicAdd(&output[output_pos2], static_cast<T>(h0lambda * w1lambda * input[pos]));
    MsAtomicAdd(&output[output_pos3], static_cast<T>(h1lambda * w0lambda * input[pos]));
    MsAtomicAdd(&output[output_pos4], static_cast<T>(h1lambda * w1lambda * input[pos]));

  }
}

template <typename T>
void CalResizeBilinearGrad(const int input_size, const T *input, const int s1, const int s2, const int s3,
                                  const int s4, T *output, const int d1, const int d2, const int d3, const int d4,
                                  bool align_corners, float h_scale, float w_scale, cudaStream_t cuda_stream) {
  int output_size = d1 * d2 * d3 * d4;
  InitZero<<<GET_BLOCKS(input_size), GET_THREADS, 0, cuda_stream>>>(output, output_size);
  ResizeBilinearGrad<<<GET_BLOCKS(input_size), GET_THREADS, 0, cuda_stream>>>(
    input_size, input, s1, s2, s3, s4, output, d1, d2, d3, d4, align_corners, h_scale, w_scale);
  return;
}

template void CalResizeBilinearGrad<float>(const int input_size, const float *input, const int s1, const int s2,
                                                  const int s3, const int s4, float *output, const int d1, const int d2,
                                                  const int d3, const int d4, bool align_corners, float h_scale,
                                                  float w_scale, cudaStream_t cuda_stream);
template void CalResizeBilinearGrad<half>(const int input_size, const half *input, const int s1, const int s2,
                                                 const int s3, const int s4, half *output, const int d1, const int d2,
                                                 const int d3, const int d4, bool align_corners, float h_scale,
                                                 float w_scale, cudaStream_t cuda_stream);
template void CalResizeBilinearGrad<int>(const int input_size, const int *input, const int s1, const int s2,
                                                const int s3, const int s4, int *output, const int d1, const int d2,
                                                const int d3, const int d4, bool align_corners, float h_scale,
                                                float w_scale, cudaStream_t cuda_stream);
