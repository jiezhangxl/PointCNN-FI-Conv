#!/usr/bin/python3
import math

num_class = 21

sample_num = 2048

batch_size = 16

num_epochs = 1024

step_val = 2000

label_weights = [0.0] * 1 + [1.0] * (num_class - 1)

learning_rate_base = 0.005
decay_steps = 5000
decay_rate = 0.8
learning_rate_min = 1e-6

weight_decay = 1e-8

jitter = 0.0
jitter_val = 0.0

rotation_range = [math.pi / 72, math.pi, math.pi / 72, 'u']
rotation_range_val = [0, 0, 0, 'u']
rotation_order = 'rxyz'

scaling_range = [0.05, 0.05, 0.05, 'g']
scaling_range_val = [0, 0, 0, 'u']

sample_num_variance = 1 // 8
sample_num_clip = 1 // 4

x = 8

xconv_param_name = ('K1', 'mm', 'sigma', 'scale','K', 'D', 'P', 'C', 'links')
xconv_params = [dict(zip(xconv_param_name, xconv_param)) for xconv_param in
                [(8 , 1, 0.002, 0.05, 8, 1, -1, 32 * x, []),
                 (12, 1, 0.005, 0.1, 12, 2, 768, 64 * x, []),
                 (16, 1, 0.04, 0.6, 16, 2, 384, 96 * x, []),
                 (16, 1, 0.05, 1, 16, 4, 128, 128 * x, [])]]    ##jie

with_global = True

xdconv_param_name = ('K', 'D', 'pts_layer_idx', 'qrs_layer_idx')
xdconv_params = [dict(zip(xdconv_param_name, xdconv_param)) for xdconv_param in
                 [(16, 4, 3, 3),
                  (16, 2, 2, 2),
                  (12, 2, 2, 1),
                  (8, 2, 1, 0)]]    ##jie


fc_param_name = ('C', 'dropout_rate')
fc_params = [dict(zip(fc_param_name, fc_param)) for fc_param in
             [(32 * x, 0.0),
              (32 * x, 0.5)]]

sampling = 'fps'

optimizer = 'adam'
epsilon = 1e-5

kernel_num = 1   ##jie
data_dim = 3

with_X_transformation = True
with_kernel_registering  = True        ##jie
with_kernel_shape_comparison = False   ##jie
with_point_transformation = False      ##jie
with_feature_transformation = False    ##jie
with_learning_feature_transformation = False     ##jie
sorting_method = None

kenel_initialization_method = 'random'     ##jie

keep_remainder = True
