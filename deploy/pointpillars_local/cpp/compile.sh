# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

mkdir -p build
cd build
rm -rf *

DEMO_NAME=main

WITH_MKL=ON
WITH_GPU=ON
USE_TENSORRT=ON

# /home/les/MyCode/PyCode/Paddle3D/inference/cu102_cudnn7.6_trt6_gcc5.4/paddle_inference/third_party/install/onnxruntime/lib/libonnxruntime.so.1.11.1
LIB_DIR=/home/les/MyCode/Paddle3D/paddle_inference
CUDNN_LIB=/usr/lib/x86_64-linux-gnu
CUDA_LIB=/usr/local/cuda/lib64
TENSORRT_ROOT=/usr/local/TensorRT-7.0.0.11
CUSTOM_OPERATOR_FILES="custom_ops/iou3d_cpu.cpp;custom_ops/iou3d_nms_api.cpp;custom_ops/iou3d_nms.cpp;custom_ops/iou3d_nms_kernel.cu"

cmake .. -DPADDLE_LIB=${LIB_DIR} \
  -DWITH_MKL=${WITH_MKL} \
  -DDEMO_NAME=${DEMO_NAME} \
  -DWITH_GPU=${WITH_GPU} \
  -DWITH_STATIC_LIB=OFF \
  -DUSE_TENSORRT=${USE_TENSORRT} \
  -DCUDNN_LIB=${CUDNN_LIB} \
  -DCUDA_LIB=${CUDA_LIB} \
  -DTENSORRT_ROOT=${TENSORRT_ROOT} \
  -DCUSTOM_OPERATOR_FILES=${CUSTOM_OPERATOR_FILES}

make -j 4
