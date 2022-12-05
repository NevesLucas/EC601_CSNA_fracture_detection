#unit test the dataset loader to make sure its working properly
import os
import kaggleDataLoader
import json
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import torchio as tio
from tqdm import tqdm
from joblib import Parallel, delayed

with open('config.json', 'r') as f:
    paths = json.load(f)

RSNA_2022_PATH    = paths["RSNA_2022_PATH"]
TRAIN_IMAGES_PREPROCESSED = f'{RSNA_2022_PATH}/train_images_cropped/'
cachedir = paths["CACHE_DIR"]
modelWeights = paths["seg_weights"]

if not os.path.exists(TRAIN_IMAGES_PREPROCESSED):
        os.makedirs(TRAIN_IMAGES_PREPROCESSED)

segModel = torch.load(modelWeights, map_location="cpu")
segModel.eval()
segResize = tio.Resize((128, 128, 200)) #resize for segmentation
classResize = tio.Resize((256,256,256))

def boundingVolume(pred,original_dims):
    #acquires the 3d bounding rectangular prism of the segmentation mask
    indices = torch.nonzero(pred)
    min_indices, min_val = indices.min(dim=0)
    max_indices, max_val = indices.max(dim=0)
    print(min_indices)
    print(max_indices)
    return (min_indices[1].item(), original_dims[0]-max_indices[1].item(),
            min_indices[2].item(), original_dims[1]-max_indices[2].item(),
            min_indices[3].item(), original_dims[2]-max_indices[3].item())

def cropData(dataElement):
    downsampled = segResize(dataElement)
    originalSize = dataElement[0].size()
    rescale = tio.Resize(originalSize)
    mask = segModel(downsampled.unsqueeze(0))
    mask = torch.argmax(mask, dim=1)
    mask = rescale(mask)
    bounding_prism = boundingVolume(mask,originalSize)
    crop = tio.Crop(bounding_prism)
    cropped = crop(dataElement)
    return classResize(cropped)

smartCrop = tio.Lambda(cropData)

HOUNSFIELD_AIR, HOUNSFIELD_BONE = -1000, 1900
clamp = tio.Clamp(out_min=HOUNSFIELD_AIR, out_max=HOUNSFIELD_BONE)
rescale = tio.RescaleIntensity(percentiles=(0.5, 99.5))
preprocess_intensity = tio.Compose([
    clamp,
    rescale,
])
normalize_orientation = tio.ToCanonical()

preprocess_spatial = tio.Compose([
    normalize_orientation])

preprocess = tio.Compose([
    preprocess_spatial,
    preprocess_intensity,
    smartCrop
])

dataLoader = kaggleDataLoader.KaggleDataLoader()

trainSet = tio.datasets.RSNACervicalSpineFracture(RSNA_2022_PATH, add_segmentations=False)

#Iterate through all the input and preprocess it

def process(subj):
    file_to_save = TRAIN_IMAGES_PREPROCESSED+subj.StudyInstanceUID+".nii"
    print(subj.StudyInstanceUID)
    if (not os.path.exists(file_to_save)):
        processed = preprocess(subj)
        processed.ct.save(file_to_save)
results = Parallel(n_jobs=24)(delayed(process)(subj) for subj in trainSet.dry_iter())