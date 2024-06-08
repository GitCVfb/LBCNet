#!/bin/bash

# create an empty folder for experimental results
mkdir -p experiments/results_demo_carla_video
mkdir -p experiments/results_demo_faster_video

cd deep_unroll_net


python inference_demo_video.py \
            --model_label='pre' \
            --results_dir=../experiments/results_demo_faster_video \
            --data_dir='../demo/Fastec' \
            --crop_sz_H=480 \
            --log_dir=../deep_unroll_weights/model_weights/fastec


python inference_demo_video.py \
            --model_label='pre' \
            --results_dir=../experiments/results_demo_carla_video \
            --data_dir='../demo/Carla' \
            --crop_sz_H=448 \
            --log_dir=../deep_unroll_weights/model_weights/carla
