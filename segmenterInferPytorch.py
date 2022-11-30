import kaggleDataLoader
import json

from joblib import Memory
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from monai.data import decollate_batch, DataLoader,Dataset,ImageDataset
from monai.metrics import DiceMetric
from monai.losses.dice import DiceLoss
from monai.networks.nets import BasicUNet
from monai.visualize import plot_2d_or_3d_image

from torchvision.ops import masks_to_boxes

import torch.cuda.amp as amp
import torchio as tio

with open('config.json', 'r') as f:
    paths = json.load(f)

def boundingVolume(pred):
    #acquires the 3d bounding rectangular prism of the segmentation mask
    temp = pred[0, :, :]
    for i in range(pred.shape[0]):
        temp = temp + pred[i, :, :]
    bbox = masks_to_boxes(temp)
    print(bbox)


cachedir = paths["CACHE_DIR"]
modelWeights = paths["seg_weights"]

root_dir="./"

if torch.cuda.is_available():
     print("GPU enabled")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = kaggleDataLoader.KaggleDataLoader()
train, val = dataset.loadDatasetAsClassifier(trainPercentage = 1.0)

model = BasicUNet(spatial_dims=3,
                  in_channels=1,
                  features=(32, 64, 128, 256, 512, 32),
                  out_channels=1).to(device)

model.load_state_dict(torch.load(modelWeights))
model.eval()

downsample = tio.Resample(1)
cropOrPad = tio.CropOrPad((128, 128, 200))

spatial_process = tio.Compose([downsample,cropOrPad])
basic_sample = train[0]

downsampled = spatial_process(basic_sample)

prediction = model(downsampled)  # just test on first image
native_prediction = prediction.apply_inverse_transform(image_interpolation='linear')

x, y, w, h, d = boundingVolume(native_prediction)
