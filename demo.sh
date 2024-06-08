#!/bin/bash

# create an empty folder for experimental results
mkdir -p experiments/results_demo_fastec
mkdir -p experiments/results_demo_carla
mkdir -p experiments/results_demo_bsrsc

cd deep_unroll_net

#Carla:  data_train/val/Carla_town03        data_train/val/Carla_town07          data_test/test/Carla_town05
#Fastec: data_train/val        data_test/test
#BSRSC: Only for generating the GS image corresponding to the middle scanline of the second RS image

python inference_demo.py \
            --model_label='pre' \
            --results_dir=../experiments/results_demo_fastec \
            --data_dir='../demo/Fastec' \
            --log_dir=../deep_unroll_weights/model_weights/fastec


python inference_demo.py \
            --model_label='pre' \
            --results_dir=../experiments/results_demo_carla \
            --data_dir='../demo/Carla' \
            --log_dir=../deep_unroll_weights/model_weights/carla

:<<!
python inference_demo.py \
            --model_label='pre' \
            --results_dir=../experiments/results_demo_bsrsc \
            --data_dir='../demo/BSRSC' \
            --log_dir=../deep_unroll_weights/model_weights/bsrsc
!
