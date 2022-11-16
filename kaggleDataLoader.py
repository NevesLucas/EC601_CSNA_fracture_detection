import copy
import pandas as pd
import pydicom
import nibabel as nib
import numpy as np
import json
import cv2
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from tqdm import tqdm

import torchio as tio

import torch
with open('config.json', 'r') as f:
    paths = json.load(f)

RSNA_2022_PATH    = paths["RSNA_2022_PATH"]
TRAIN_IMAGES_PATH = f'{RSNA_2022_PATH}/train_images'
TEST_IMAGES_PATH  = f'{RSNA_2022_PATH}/test_images'


target_cols = ['C1', 'C2', 'C3',
               'C4', 'C5', 'C6', 'C7',
               'patient_overall']

revert_dict = {
    '1.2.826.0.1.3680043.1363',
    '1.2.826.0.1.3680043.20120',
    '1.2.826.0.1.3680043.2243',
    '1.2.826.0.1.3680043.24606',
    '1.2.826.0.1.3680043.32071',
    '1.2.826.0.1.3680043.20574'
}

def loadDicom(path):
    img = pydicom.dcmread(path)
    img.PhotometricInterpretation = 'YBR_FULL'
    data = img.pixel_array
    data = data - np.min(data)
    if np.max(data) != 0:
        data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return cv2.cvtColor(data, cv2.COLOR_GRAY2RGB), img


class KaggleDataLoader:

    def __init__(self):
        # Load metadata

        self.trainImagePath = TRAIN_IMAGES_PATH
        self.segPath        = os.path.join(RSNA_2022_PATH,          "segmentations")
        self.trainDf        = pd.read_csv(os.path.join(RSNA_2022_PATH, "train.csv"))
        self.trainBbox      = pd.read_csv(os.path.join(RSNA_2022_PATH, "train_bounding_boxes.csv"))
        self.testDf         = pd.read_csv(os.path.join(RSNA_2022_PATH, "test.csv"))
        self.ss             = pd.read_csv(os.path.join(RSNA_2022_PATH, "sample_submission.csv"))

    ## Dataset generator functions
    def loadDatasetAsClassifier(self, trainPercentage=0.90, train_aug=None):
        """
        prepare full dataset for training
        """
        HOUNSFIELD_AIR, HOUNSFIELD_BONE = -1000, 1900
        clamp = tio.Clamp(out_min=HOUNSFIELD_AIR, out_max=HOUNSFIELD_BONE)
        rescale = tio.RescaleIntensity(percentiles=(0.5, 99.5))
        preprocess_intensity = tio.Compose([
            clamp,
            rescale,
        ])
        normalize_orientation = tio.ToCanonical()
        downsample = tio.Resample(1)

        cropOrPad = tio.CropOrPad((130,130,200))
        preprocess_spatial = tio.Compose([
            normalize_orientation,
            downsample,
            cropOrPad,
        ])
        preprocess = tio.Compose([
            preprocess_spatial,
            preprocess_intensity,
        ])

        trainSet = tio.datasets.RSNACervicalSpineFracture(RSNA_2022_PATH)
        #strip out bad entries
        trainSet = tio.data.SubjectsDataset(list(filter( lambda subject : subject.StudyInstanceUID not in revert_dict, trainSet.dry_iter())))
        num_subjects = len(trainSet)
        num_train = int(trainPercentage*num_subjects)
        num_val = num_subjects - num_train
        train_set, val_set = torch.utils.data.random_split(trainSet,[num_train,num_val])
        val_set = copy.deepcopy(val_set)
        train_set.dataset.set_transform(preprocess)
        val_set.dataset.set_transform(preprocess)
        if train_aug is not None:
            val_set = copy.deepcopy(val_set)
            augment = tio.Compose([
                preprocess,
                train_aug
            ])
            train_set.dataset.set_transform(augment)
            val_set.dataset.set_transform(preprocess)

        return train_set, val_set


    def loadDatasetAsDetector(self):
        """
        prepare full dataset for training
        """

    def loadDatasetAsSegmentor(self, trainPercentage=0.90, train_aug=None):
        """
        prepare full dataset for training
        """

        HOUNSFIELD_AIR, HOUNSFIELD_BONE = -1000, 1900
        clamp = tio.Clamp(out_min=HOUNSFIELD_AIR, out_max=HOUNSFIELD_BONE)
        rescale = tio.RescaleIntensity(percentiles=(0.5, 99.5))
        preprocess_intensity = tio.Compose([
            clamp,
            rescale,
        ])
        normalize_orientation = tio.ToCanonical()
        transform = tio.Resample('ct')
        downsample = tio.Resample(1)
        cropOrPad = tio.CropOrPad((128,128,200))
        preprocess_spatial = tio.Compose([
            normalize_orientation,
            downsample,
            transform,
            cropOrPad,
        ])
        sequential = tio.SequentialLabels()
        remapping = {2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1}
        remap_mask = tio.RemapLabels(remapping)

        preprocess = tio.Compose([
            sequential,
            remap_mask,
            preprocess_spatial,
            preprocess_intensity
        ])
        trainSet = tio.datasets.RSNACervicalSpineFracture(RSNA_2022_PATH, add_segmentations=True)
        trainSet = tio.data.SubjectsDataset(list(filter( lambda seg : 'seg' in seg, trainSet.dry_iter())))

        #strip out bad entries
        trainSet = tio.data.SubjectsDataset(list(filter( lambda subject : subject.StudyInstanceUID not in revert_dict, trainSet.dry_iter())))
        num_subjects = len(trainSet)
        num_train = int(trainPercentage*num_subjects)
        num_val = num_subjects - num_train
        train_set, val_set = torch.utils.data.random_split(trainSet,[num_train,num_val])
        train_set.dataset.set_transform(preprocess)
        if train_aug is not None:
            val_set = copy.deepcopy(val_set)
            augment = tio.Compose([
                preprocess,
                train_aug
            ])
            train_set.dataset.set_transform(augment)
            val_set.dataset.set_transform(preprocess)
        return train_set, val_set
