# stdlib
import os

# PyTorch
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

# 3rd party packages
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# modules
from net.models.vgg import *
from net.models.alexnet import *


def vis_square(image):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
    image = image.numpy()

    # normalize data for display
    data = (image - image.min()) / (image.max() - image.min())

    count = 0
    for img in data:
        count += 1
        if(count>30):
            print(img*255)
            im = Image.fromarray(img*255).convert("L")
            im.show()
            break
            im.save('filter'+str(count)+".jpg")


def vis_single_square(image,savedir):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
    image = image.numpy()

    # normalize data for display
    data = (image - image.min()) / (image.max() - image.min())
    data = data[0,0,:,:]

    im = Image.fromarray(data*255).convert("L")
    im.show()
    im.save(savedir)



def load_valid(model, pretrained_file, skip_list=None):

    pretrained_dict = torch.load(pretrained_file)
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    if skip_list is not None:
        pretrained_dict1 = {k: v for k, v in pretrained_dict.items() if k in model_dict and k not in skip_list }
    else:
        pretrained_dict1 = pretrained_dict
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict1)
    model.load_state_dict(model_dict)
    return model

if __name__ == "__main__":
    pass