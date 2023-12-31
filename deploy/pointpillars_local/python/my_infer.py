#!/usr/bin/python 
# -*- coding: utf-8 -*-
# @time    :09/03/23
# @Author  :cxy
# @file    :my_infer.py


# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import time
import numba
import numpy as np
import paddle
from paddle.inference import Config, create_predictor

from paddle3d.ops.iou3d_nms_cuda import nms_gpu


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_file",
        type=str,
        help="Model filename, Specify this when your model is a combined model.")
        # required=True)
    parser.add_argument(
        "--params_file",
        type=str,
        help=
        "Parameter filename, Specify this when your model is a combined model.")
        # required=True)
    # parser.add_argument(
    #     '--lidar_file', type=str, help='The lidar path.', required=True)
    parser.add_argument(
        "--num_point_dim",
        type=int,
        default=4,
        help="Dimension of a point in the lidar file.")
    parser.add_argument(
        "--point_cloud_range",
        dest='point_cloud_range',
        nargs='+',
        help="Range of point cloud for voxelize operation.",
        type=float,
        default=None)
    parser.add_argument(
        "--voxel_size",
        dest='voxel_size',
        nargs='+',
        help="Size of voxels for voxelize operation.",
        type=float,
        default=None)
    parser.add_argument(
        "--max_points_in_voxel",
        type=int,
        default=100,
        help="Maximum number of points in a voxel.")
    parser.add_argument(
        "--max_voxel_num",
        type=int,
        default=12000,
        help="Maximum number of voxels.")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU card id.")
    parser.add_argument(
        "--use_trt",
        type=int,
        default=0,
        help="Whether to use tensorrt to accelerate when using gpu.")
    parser.add_argument(
        "--trt_precision",
        type=int,
        default=0,
        help="Precision type of tensorrt, 0: kFloat32, 1: kHalf.")
    parser.add_argument(
        "--trt_use_static",
        type=int,
        default=0,
        help="Whether to load the tensorrt graph optimization from a disk path."
    )
    parser.add_argument(
        "--trt_static_dir",
        type=str,
        help="Path of a tensorrt graph optimization directory.")
    parser.add_argument(
        "--collect_shape_info",
        type=int,
        default=0,
        help="Whether to collect dynamic shape before using tensorrt.")
    parser.add_argument(
        "--dynamic_shape_file",
        type=str,
        default="",
        help="Path of a dynamic shape file for tensorrt.")

    return parser.parse_args()


def read_point(file_path, num_point_dim):
    points = np.fromfile(file_path, np.float32).reshape(-1, num_point_dim)
    points = points[:, :4]
    return points


@numba.jit(nopython=True)
def _points_to_voxel(points, voxel_size, point_cloud_range, grid_size, voxels,
                     coords, num_points_per_voxel, grid_idx_to_voxel_idx,
                     max_points_in_voxel, max_voxel_num):
    num_voxels = 0
    num_points = points.shape[0]
    # x, y, z
    coord = np.zeros(shape=(3, ), dtype=np.int32)

    for point_idx in range(num_points):
        outside = False
        for i in range(3):
            coord[i] = np.floor(
                (points[point_idx, i] - point_cloud_range[i]) / voxel_size[i])
            if coord[i] < 0 or coord[i] >= grid_size[i]:
                outside = True
                break
        if outside:
            continue
        voxel_idx = grid_idx_to_voxel_idx[coord[2], coord[1], coord[0]]
        if voxel_idx == -1:
            voxel_idx = num_voxels
            if num_voxels >= max_voxel_num:
                continue
            num_voxels += 1
            grid_idx_to_voxel_idx[coord[2], coord[1], coord[0]] = voxel_idx
            coords[voxel_idx, 0:3] = coord[::-1]
        curr_num_point = num_points_per_voxel[voxel_idx]
        if curr_num_point < max_points_in_voxel:
            voxels[voxel_idx, curr_num_point] = points[point_idx]
            num_points_per_voxel[voxel_idx] = curr_num_point + 1

    return num_voxels


def hardvoxelize(points, point_cloud_range, voxel_size, max_points_in_voxel,
                 max_voxel_num):
    num_points, num_point_dim = points.shape[0:2]
    point_cloud_range = np.array(point_cloud_range)
    voxel_size = np.array(voxel_size)
    voxels = np.zeros((max_voxel_num, max_points_in_voxel, num_point_dim),
                      dtype=points.dtype)
    coords = np.zeros((max_voxel_num, 3), dtype=np.int32)
    num_points_per_voxel = np.zeros((max_voxel_num, ), dtype=np.int32)
    grid_size = np.round((point_cloud_range[3:6] - point_cloud_range[0:3]) /
                         voxel_size).astype('int32')

    grid_size_x, grid_size_y, grid_size_z = grid_size

    grid_idx_to_voxel_idx = np.full((grid_size_z, grid_size_y, grid_size_x),
                                    -1,
                                    dtype=np.int32)

    num_voxels = _points_to_voxel(points, voxel_size, point_cloud_range,
                                  grid_size, voxels, coords,
                                  num_points_per_voxel, grid_idx_to_voxel_idx,
                                  max_points_in_voxel, max_voxel_num)

    voxels = voxels[:num_voxels]
    coords = coords[:num_voxels]
    num_points_per_voxel = num_points_per_voxel[:num_voxels]

    return voxels, coords, num_points_per_voxel


def preprocess(file_path, num_point_dim, point_cloud_range, voxel_size,
               max_points_in_voxel, max_voxel_num):
    points = read_point(file_path, num_point_dim)
    voxels, coords, num_points_per_voxel = hardvoxelize(
        points, point_cloud_range, voxel_size, max_points_in_voxel,
        max_voxel_num)

    return voxels, coords, num_points_per_voxel


def init_predictor(model_file,
                   params_file,
                   gpu_id=0,
                   use_trt=False,
                   trt_precision=0,
                   trt_use_static=False,
                   trt_static_dir=None,
                   collect_shape_info=False,
                   dynamic_shape_file=None):
    config = Config(model_file, params_file)
    config.enable_memory_optim()
    config.enable_use_gpu(1000, gpu_id)
    if use_trt:
        precision_mode = paddle.inference.PrecisionType.Float32
        if trt_precision == 1:
            precision_mode = paddle.inference.PrecisionType.Half
        config.enable_tensorrt_engine(
            workspace_size=1 << 20,
            max_batch_size=1,
            min_subgraph_size=10,
            precision_mode=precision_mode,
            use_static=trt_use_static,
            use_calib_mode=False)
        if collect_shape_info:
            config.collect_shape_range_info(dynamic_shape_file)
        else:
            config.enable_tuned_tensorrt_dynamic_shape(dynamic_shape_file, True)
        if trt_use_static:
            config.set_optim_cache_dir(trt_static_dir)

    predictor = create_predictor(config)
    return predictor


def parse_result(box3d_lidar, label_preds, scores, save_path, save_path_2):
    num_bbox3d, bbox3d_dims = box3d_lidar.shape
    for box_idx in range(num_bbox3d):
        # filter fake results: score = -1
        if scores[box_idx] < 0:
            # print("result is none.")
            continue
        if scores[box_idx] > 0.4:
            if bbox3d_dims == 7:
                # cxy 20230424
                # box3d_lidar[box_idx, 2] = box_idx[box_idx, 2] + box3d_lidar[box_idx, 5]/2
                print("Score: {} Label: {} Box(x_c, y_c, z_c, w, l, h, -rot): {} {} {} {} {} {} {}"
                            .format(scores[box_idx], label_preds[box_idx],
                            box3d_lidar[box_idx, 0], box3d_lidar[box_idx, 1],
                            box3d_lidar[box_idx, 2], box3d_lidar[box_idx, 3],
                            box3d_lidar[box_idx, 4], box3d_lidar[box_idx, 5],
                            box3d_lidar[box_idx, 6]))
                list_result = [box3d_lidar[box_idx, 0], box3d_lidar[box_idx, 1], box3d_lidar[box_idx, 2],
                               box3d_lidar[box_idx, 3], box3d_lidar[box_idx, 4], box3d_lidar[box_idx, 5],
                               box3d_lidar[box_idx, 6]]

                with open(save_path, "a", encoding="utf-8") as fa:
                    fa.write(str(list_result)[1:-1] + "\n")

                list_result_2 = [label_preds[box_idx], box3d_lidar[box_idx, 0], box3d_lidar[box_idx, 1],
                                box3d_lidar[box_idx, 2], box3d_lidar[box_idx, 3], box3d_lidar[box_idx, 4],
                                box3d_lidar[box_idx, 5], box3d_lidar[box_idx, 6], scores[box_idx]]

                with open(save_path_2, "a", encoding="utf-8") as fa:
                    fa.write(str(list_result_2)[1:-1] + "\n")


def run(predictor, voxels, coords, num_points_per_voxel):
    input_names = predictor.get_input_names()
    for i, name in enumerate(input_names):
        input_tensor = predictor.get_input_handle(name)
        if name == "voxels":
            input_tensor.reshape(voxels.shape)
            input_tensor.copy_from_cpu(voxels.copy())
        elif name == "coords":
            input_tensor.reshape(coords.shape)
            input_tensor.copy_from_cpu(coords.copy())
        elif name == "num_points_per_voxel":
            input_tensor.reshape(num_points_per_voxel.shape)
            input_tensor.copy_from_cpu(num_points_per_voxel.copy())

    # do the inference
    predictor.run()

    # get out data from output tensor
    output_names = predictor.get_output_names()
    for i, name in enumerate(output_names):
        output_tensor = predictor.get_output_handle(name)
        if i == 0:
            box3d_lidar = output_tensor.copy_to_cpu()
        elif i == 1:
            label_preds = output_tensor.copy_to_cpu()
        elif i == 2:
            scores = output_tensor.copy_to_cpu()
    return box3d_lidar, label_preds, scores


def main(args):

    if not os.path.exists(predict_result_dir):
        os.mkdir(predict_result_dir)
    if not os.path.exists(predict_result_dir2):
        os.mkdir(predict_result_dir2)
    predictor = init_predictor(args.model_file, args.params_file, args.gpu_id,
                               args.use_trt, args.trt_precision,
                               args.trt_use_static, args.trt_static_dir,
                               args.collect_shape_info, args.dynamic_shape_file)

    # for bin_name in os.listdir(test_root_dir):
    for bin_name in range(start_frame, end_frame+1):
        bin_name = (6 - len(str(bin_name))) * str(0) + str(bin_name) + ".bin"

        args.lidar_file = os.path.join(test_root_dir, bin_name)

        if not os.path.exists(args.lidar_file):
            continue
        print("test_bin_name:", args.lidar_file)
        test_file_name = os.path.join(predict_result_dir, bin_name + ".txt")
        test_file_name2 = os.path.join(predict_result_dir2, bin_name + ".txt")

        start_time = time.time()
        voxels, coords, num_points_per_voxel = preprocess(
            args.lidar_file, args.num_point_dim, args.point_cloud_range,
            args.voxel_size, args.max_points_in_voxel, args.max_voxel_num)
        box3d_lidar, label_preds, scores = run(predictor, voxels, coords,
                                               num_points_per_voxel)
        end_time = time.time()
        print("total cost time: ", end_time - start_time)
        parse_result(box3d_lidar, label_preds, scores, test_file_name, test_file_name2)


if __name__ == '__main__':
    args = parse_args()

    start_frame = 0
    end_frame = 20000
    mode_root_dir = r"/home/les/MyCode/PyCode/pointpilliars_for_docker/outputs/pointpillars_airplane_new_sample/iter_80000"
    args.model_file = os.path.join(mode_root_dir, "pointpillars.pdmodel")
    args.params_file = os.path.join(mode_root_dir, "pointpillars.pdiparams")

    root_dir = r"/home/les/MyCode/PyCode/pointpillisars_local/datasets/KITTI/training"
    test_root_dir = os.path.join(root_dir, "val")   # velodyne  test
    predict_result_dir = os.path.join(root_dir, "test_result_20230525")
    predict_result_dir2 = os.path.join(root_dir, "test_result_20230525_2")

    main(args)


# python infer.py \
#   --model_file /path/to/pointpillars.pdmodel \
#   --params_file /path/to/pointpillars.pdiparams \
#   --lidar_file /path/to/lidar.bin \
#   --point_cloud_range 0 -39.68 -3 69.12 39.68 1 \
#   --voxel_size .16 .16 4 \
#   --max_points_in_voxel 32 \
#   --max_voxel_num 40000

# python my_infer.py --point_cloud_range 0 -39.68 -3 69.12 39.68 1 --voxel_size .16 .16 4 --max_points_in_voxel 32  --max_voxel_num 40000

# python my_infer.py --point_cloud_range 0 -79.36 -6 207.36 79.36 6 --voxel_size .16 .16 12  --max_points_in_voxel 32  --max_voxel_num 40000
# [ 0, -59.52, -3, 142.08, 59.52, 3 ]
# [ 0, -59.52, -6, 138.24, 59.52, 6 ] [ 0, -39.68, -6, 138.24, 39.68, 6 ]