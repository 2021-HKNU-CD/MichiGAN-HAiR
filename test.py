import os
from collections import OrderedDict

import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util.util import tensor2im, tensor2label, blend_image
from util import html
from data.base_dataset import single_inference_dataLoad
from PIL import Image
import torch
import math
import numpy as np
import torch.nn as nn
import cv2
from opt import opt

opt.inference_orient_name = "99999"
opt.inference_ref_name = "67172"
opt.inference_tag_name = "99999"
model = Pix2PixModel(opt)
model.eval()
# read data
data = single_inference_dataLoad(opt)
# forward
# generated <- generated_fake_image
generated = model(data, mode='inference')
img_path = data['path']
print('process image... %s' % img_path)

fake_image = tensor2im(generated[0])
if opt.add_feat_zeros or opt.add_zeros:
    th = opt.add_th
    H, W = opt.crop_size, opt.crop_size
    fake_image_tmp =\
        fake_image[int(th/2):int(th/2)+H, int(th/2):int(th/2)+W, :]
    print(len(fake_image))
    fake_image = fake_image_tmp

fake_image_np = fake_image.copy()
fake_image = Image.fromarray(np.uint8(fake_image))

if opt.use_ig:
    fake_image.save('./inference_samples/inpaint_fake_image.jpg')
else:
    fake_image.save('./inference_samples/fake_image.jpg')
