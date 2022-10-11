import torch
import torch.nn as nn
import os
import numpy as np
import tqdm
import json
from AFSD.common import videotransforms
from AFSD.common.anet_dataset import get_video_info, load_json
from AFSD.anet.BDNet import BDNet
from AFSD.common.segment_utils import softnms_v2
from AFSD.common.config import config

import multiprocessing as mp
import threading
from thop import profile

# num_classes = config['dataset']['num_classes']
# conf_thresh = config['testing']['conf_thresh']
# top_k = config['testing']['top_k']
# nms_thresh = config['testing']['nms_thresh']
# nms_sigma = config['testing']['nms_sigma']
# clip_length = config['dataset']['testing']['clip_length']
# stride = config['dataset']['testing']['clip_stride']
# checkpoint_path = config['testing']['checkpoint_path']
# json_name = config['testing']['output_json']
# output_path = config['testing']['output_path']
# softmax_func = True
# if not os.path.exists(output_path):
#     os.makedirs(output_path)
# fusion = config['testing']['fusion']

# # getting path for fusion
# rgb_data_path = config['testing'].get('rgb_data_path',
#                                       './datasets/thumos14/test_npy/')
# flow_data_path = config['testing'].get('flow_data_path',
#                                        './datasets/thumos14/test_flow_npy/')
rgb_checkpoint_path = 'models/anet/checkpoint-10.ckpt'
flow_checkpoint_path = 'models/anet_flow/checkpoint-6.ckpt'

if __name__ == '__main__':
    rgb_net = BDNet(in_channels=3, training=False)
    flow_net = BDNet(in_channels=2, training=False)
    rgb_net.load_state_dict(torch.load(rgb_checkpoint_path))
    flow_net.load_state_dict(torch.load(flow_checkpoint_path))
    device = torch.device('cuda:0')
    rgb_net=rgb_net.to(device)
    flow_net = flow_net.to(device)
    # rgb_net.eval().cuda()
    # flow_net.eval().cuda()

    inputRGB = torch.randn(1, 3,768,96,96)  # 放进和model相同的GPU中
    inputFLOW = torch.randn(1, 2,768,96,96)  # 放进和model相同的GPU中
    inputRGB=inputRGB.to(device)
    inputFLOW=inputFLOW.to(device)

    flops, params = profile(rgb_net, inputs=(inputRGB,))
    print("%s ------- params: %.2fMB ------- flops: %.2fG" % (
    rgb_net, params / (1000 ** 2), flops / (1000 ** 3)))  # 这里除以1000的平方，是为了化成M的单位

    # flops, params = profile(flow_net, inputs=(inputFLOW,), verbose=True)
    # print("%s ------- params: %.2fMB ------- flops: %.2fG" % (
    # flow_net, params / (1000 ** 2), flops / (1000 ** 3)))  # 这里除以1000的平方，是为了化成M的单位



