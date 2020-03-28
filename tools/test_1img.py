# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

import os
import pprint
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import lib.models as models
from lib.config import config, update_config
from lib.utils import utils
from lib.datasets import get_dataset
from lib.core import function

################################################################
import cv2
import numpy as np

def imread_RGB(filename):
    # read image
    img = cv2.imread(filename)
    # convert color from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # return RGB image
    return img

# [specifications rough]
#   - when scale and pixel are input at the same time.
#   - if there is no value in either height or width, resize with keeping aspect ratio.
def imresize(img, height_scale=None, width_scale=None, 
                  height_pixel=None, width_pixel=None, 
                  clip_0_255=True, nearest=False):
    # simultaneous input of scale and pixel
    if (((height_scale is not None) | (width_scale is not None)) &  
        ((height_pixel is not None) | (width_pixel is not None))):
        # warning
        warnings.warn('adopted scale (reject pixel)')
    # input scale
    if ((height_scale is not None) | (width_scale is not None)): 
        # input only height
        if ((height_scale is not None) & (width_scale is None)):
            # adjust width to height
            width_scale = height_scale
        # input only width
        if ((height_scale is None) & (width_scale is not None)):
            # adjust height to width
            height_scale = width_scale
        # calc height pixel and width pixel of resized image
        height_pixel = int(np.shape(img)[0] * height_scale)
        width_pixel  = int(np.shape(img)[1] * width_scale)
    # input pixel
    else:
        # input only height
        if ((height_pixel is not None) & (width_pixel is None)):
            # adjust width to height
            width_pixel = int(np.round(height_pixel * (np.shape(img)[1] / np.shape(img)[0])))
        # input only width
        if ((height_pixel is None) & (width_pixel is not None)):
            # adjust height to width
            height_pixel = int(np.round(width_pixel * (np.shape(img)[0] / np.shape(img)[1])))
    # resize
    if (nearest == True):
        img = cv2.resize(img, (width_pixel, height_pixel), 
                              interpolation=cv2.INTER_NEAREST) # for segmentation map
    else:
        img = cv2.resize(img, (width_pixel, height_pixel), 
                              interpolation=cv2.INTER_CUBIC)
    # adjust brightness between 0 and 255
    if (clip_0_255):
        img = np.clip(img, 0, 255)
    #     
    return img

from torchsummary import summary
from lib.core.evaluation import decode_preds
import matplotlib.pyplot as plt
################################################################


def parse_args():

    parser = argparse.ArgumentParser(description='Train Face Alignment')

    parser.add_argument('--cfg', help='experiment configuration filename',
                        required=True, type=str)
    parser.add_argument('--model-file', help='model parameters', required=True, type=str)

    args = parser.parse_args()
    update_config(config, args)
    return args


def main():

    args = parse_args()

    logger, final_output_dir, tb_log_dir = \
        utils.create_logger(config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    config.defrost()
    config.MODEL.INIT_WEIGHTS = False
    config.freeze()
    model = models.get_face_alignment_net(config)

    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    # load model
    state_dict = torch.load(args.model_file)
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict)
    else:
        # print('state_dict = ')
        # print(state_dict)
        # model.module.load_state_dict(state_dict)
        model.load_state_dict(state_dict)

    ############################################################
    # check keras-like model summary using torchsummary [https://github.com/usuyama/pytorch-unet]
    print(model)
    # summary(model, input_size=(3, 256, 256))
    
    # loop of image file
    for file_i in (np.arange(5) + 1):
        # set filename
        filename_tmp = ('./test_%d.png' % file_i)
        # image read and resize
        img = imread_RGB(filename_tmp)
        img = imresize(img, height_pixel=256, 
                            width_pixel=256)
        # standardization
        X           = img.copy()
        X           = X.astype(np.float32)
        X          /= 255
        X[:, :, 0] -= 0.485
        X[:, :, 1] -= 0.456
        X[:, :, 2] -= 0.406
        X[:, :, 0] /= 0.229
        X[:, :, 1] /= 0.224
        X[:, :, 2] /= 0.225
        # adjust shape
        X = X[np.newaxis, :, :, :]
        X = X.transpose(0, 3, 1, 2)
        print('np.shape(X) = ', end='')
        print(np.shape(X))
        # transport to GPU (or stay on CPU)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X = torch.Tensor(X).to(device) 
        # set evaluation mode to model
        model.eval() 
        # not need grad
        with torch.no_grad():
            output = model(X)
            score_map = output.data.cpu()
            y_hat = decode_preds(score_map, [[128.0, 128.0]], [1.2], [64, 64])
        # transport to CPU
        y_hat = y_hat.to('cpu').numpy()
        print('np.shape(y_hat) = ', end='')
        print(np.shape(y_hat))
        print(y_hat)
        
        # ランドマーク描画
        for (x, y) in y_hat[0]:
            cv2.circle(img, (x, y), radius=3, color=(200, 100, 0), thickness=3) # 大体の肌色の補色
        # グラフの描画先の準備
        plt.figure()
        # 描画
        plt.imshow(img)
        # save as png
        plt.savefig(filename_tmp.replace('.png', '_output.png')) # -----(2)
    ############################################################    


if __name__ == '__main__':
    main()

