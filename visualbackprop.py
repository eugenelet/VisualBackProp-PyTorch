# stdlib
import os

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.transforms as transforms

# 3rd party packages
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# modules
from net.models.vgg import *
from net.models.alexnet import *
# from net.utility.tools import *

import math
import pdb

maps = []
layers=[]
hooks=[]


FEAT_KEEP = 30  # Feature Maps to show
FEAT_SIZE = 224  # Size of feature maps to show
FEAT_MAPS_DIR = 'feat_maps'  # dir. to save feat maps
VBP_DIR = 'VBP_results'


def normalize_gamma(image, gamma=1.0):
	# normalize data for display
	image = (image - image.min()) / (image.max() - image.min())
	invGamma = 1.0 / gamma
	image = (image ** invGamma)  * 255
	return image.astype("uint8")


def visual_feature(self, input, output):
	# The hook function that show you the feature maps while forward propagate

	vis_square(output.data[0,:])

def save_feature_maps(self,input,output):
	# The hook function that saves feature maps while forward propagate

	map = output.data
	maps.append(map)


def add_hook(net,layer_name,func):
	'''
	Add a hook function in the layers you specified.
	Hook will be called during forward propagate at the layer you specified.

	:param net: The model you defined
	:param layer_name: Specify which layer you want to hook, currently you can hook 'all', 'maxpool', 'relu'
	:param func: Specify which hook function you want to hook while forward propagate
	:return: this function will return the model that hooked the function you specified in specific layer
	'''

	if layer_name=='maxpool':
		for m in net.features:
			if isinstance(m, nn.MaxPool2d):
				m.register_forward_hook(func)
		return net

	if layer_name == 'relu':
		for index, m in enumerate(net.features):
			if isinstance(m, nn.ReLU):
				type_name = str(type(m)).replace("<'", '').replace("'>", '').split('.')[-1]
				name = 'features' + '-' + str(index) + '-' + type_name
				hook = m.register_forward_hook(func)
				layers.append((name, m))
				hooks.append(hook)
		return net

	if layer_name == 'all':
		for index, m in enumerate(net.features):
			type_name = str(type(m)).replace("<'", '').replace("'>", '').split('.')[-1]
			name = 'features' + '-' + str(index) + '-' + type_name
			hook = m.register_forward_hook(func)
			layers.append((name, m))
			hooks.append(hook)
		return net


def visualbackprop(layers,maps):

	'''
	:param layers: the saved layers
	:param maps: the saved maps
	:return: return the final mask
	'''

	num_layers = len(maps)
	avgs = []
	mask = None
	ups  = []

	upSample = nn.Upsample(scale_factor=2)

	for n in range(num_layers-1,0,-1):
		cur_layer=layers[n][1]
		if type(cur_layer) in [torch.nn.MaxPool2d]:
			print(layers[n][0])
			##########################
			# Get and set attributes #
			##########################
			relu = maps[n-1]
			conv = maps[n-2]

			###########################################
			# Average filters and multiply pixel-wise #
			###########################################

			# Average filters
			avg = relu.mean(dim=1)
			avg = avg.unsqueeze(0)
			avgs.append(avg)

			if mask is not None:
				mask = upSample(Variable(mask)).data
				mask = mask * avg
			else:
				mask = avg

			# upsampling : see http://pytorch.org/docs/nn.html#convtranspose2d
			weight = Variable(torch.ones(1, 1, 3, 3))
			up = F.conv_transpose2d(Variable(mask), weight, stride=1, padding=1)
			mask = up.data
			ups.append(mask)

	return ups






def plotFeatMaps(layers,maps):

	'''
	:param layers: the saved layers
	:param maps: the saved maps
	:return: top feat. maps of relu layers
	'''

	num_layers = len(maps)
	feat_collection = []
	# Show top FEAT_KEEP feature maps (after ReLU) starting from bottom layers
	for n in range(num_layers):
		cur_layer=layers[n][1]
		if type(cur_layer) in [torch.nn.MaxPool2d]:
			##########################
			# Get and set attributes #
			##########################
			relu = maps[n-1]

			###########################################
			# Sort Feat Maps based on energy of F.M. #
			###########################################
			feat_energy = []
			# Get energy of each channel
			for channel_n in range(relu.shape[1]):
				feat_energy.append(np.sum(relu[0][channel_n].numpy()))
			feat_energy = np.array(feat_energy)
			# Sort energy
			feat_rank = np.argsort(feat_energy)[::-1]

			# Empty background
			back_len = int(math.ceil(math.sqrt(FEAT_SIZE * FEAT_SIZE * FEAT_KEEP * 2)))
			feat = np.zeros((back_len, back_len))
			col = 0
			row = 0
			for feat_n in range(FEAT_KEEP):
				if col*FEAT_SIZE + FEAT_SIZE < back_len:
					feat[row*FEAT_SIZE:row*FEAT_SIZE + FEAT_SIZE, col*FEAT_SIZE:col*FEAT_SIZE + FEAT_SIZE] =\
						cv2.resize(normalize_gamma(relu[0][feat_rank[feat_n]].numpy(), 0.1), (FEAT_SIZE,FEAT_SIZE))
					col = col + 1
				else:
					row = row + 1
					col = 0
					feat[row*FEAT_SIZE:row*FEAT_SIZE + FEAT_SIZE, col*FEAT_SIZE:col*FEAT_SIZE + FEAT_SIZE] =\
						cv2.resize(normalize_gamma(relu[0][feat_rank[feat_n]].numpy(), 0.1), (FEAT_SIZE,FEAT_SIZE))
					col = col + 1

			feat_collection.append(feat)

	return feat_collection


# Show VBP Result
def show_VBP(label, image):
	"""Take an array of shape (n, height, width) or (n, height, width, 3)
	   and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
	image = image.numpy()

	# normalize data for display
	data = (image - image.min()) / (image.max() - image.min())
	data = data[0,0,:,:]
	data = cv2.resize(data, (224,224))
	data = (data*255).astype("uint8")
	# cv2.imwrite(label, data)
	cv2.imshow(label, data)


# Save VBP Result
def save_VBP(label, image):
	image = image.numpy()

	# normalize data for display
	data = (image - image.min()) / (image.max() - image.min())
	data = data[0,0,:,:]
	data = cv2.resize(data, (224,224))
	data = (data*255).astype("uint8")
	# cv2.imwrite(label, data)
	cv2.imwrite(label, data)


def overlay(image, mask):
	# normalize data for display
	mask = (mask - mask.min()) / (mask.max() - mask.min())
	mask = mask[0,0,:,:]
	mask = cv2.resize(mask, (224,224))
	mask = (mask*255).astype("uint8")
	# pdb.set_trace()
	# assert image.shape == mask.shape, "image %r and mask %r must be of same shape" % (image.shape, mask.shape)
	# if image[:,:,2] + mask > 255:
		# image[:,:,2] = image[:,:,2] + mask
	# else:
	image[:,:,2] = cv2.add(image[:,:,2], mask)

	return image

if __name__ == "__main__":

	'''
	Load image, resize the image to 224 x 224, then transfer the loaded image numpy array.
	The transfer included 1.ArrayToTensor and 2.Normalization.
	After image transformation, we need to define the model.
	The model used here is VGG19, you can choose whatever model you like.
	After the model is defined, the pre-trained model is loaded.
	
	Before we forward the image through VGG, we need to define our hook first.
	The function "add_hook" provide you an easy way to add hook to layer.
	You have to specify: 
		1. the model you want to hook 
		2. the layer you want to hook
		3. the function you want to hook
	
	Since I want to save the feature maps to a list.
	I create "save_image_maps" as my hook function
	This function will help me to extract the layer output.
	After the outputs are extracted, I want to output the extracted feature maps to image.
	So by calling "save_image_maps", we call save all the maps as image locally.
	'''


	BASE_DIR = os.path.dirname(os.path.abspath(os.path.dirname('__dir__')))
	IMG_DIR = './image'
	# MODEL_DIR = BASE_DIR + '/pretrained_model'
	IMG_NAME = 'cat2.jpg'


	image = cv2.imread(IMG_DIR+'/'+IMG_NAME)
	image = cv2.resize(image, (224, 224))

	im = Image.fromarray(image)
	im.save('./image/resized_' +IMG_NAME)

	loader = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
	])
	img = loader(image)
	img = img.contiguous().view(1, 3, 224, 224)
	model = vgg19()

	#load_valid(model, MODEL_DIR + "/vgg19-dcbb9e9d.pth", None)
	x = Variable(img)


	add_hook(model,'all',save_feature_maps)
	logits, probs = model.forward(x)

	feat_collection = plotFeatMaps(layers, maps)

	# Save Feature Maps
	for i in range(len(feat_collection)):
		cv2.imwrite(FEAT_MAPS_DIR + '/feat_' + str(i) + '_' + IMG_NAME, feat_collection[i] * 255)
	masks = visualbackprop(layers, maps)

	mask_num = len(masks)

	cv2.imshow('ori', image)
	cv2.moveWindow('ori', 50, 50)
	for i in range(mask_num):
		save_VBP(VBP_DIR + '/out_' + str(i) + '_' + IMG_NAME, masks[i])
		show_VBP('vbp_' + str(i) + '.png', masks[i])
		cv2.moveWindow('vbp_' + str(i) + '.png', i*30 + 100, i*30 + 100)


	overlay_img = overlay(image, masks[mask_num - 1].numpy())
	cv2.imshow('overlay', overlay_img)
	cv2.moveWindow('overlay', 200, 200)
	cv2.imwrite('overlay.png', overlay_img)
	cv2.waitKey(0)
