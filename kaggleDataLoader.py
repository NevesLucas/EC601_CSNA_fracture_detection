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
tio.Subject.relative_attribute_tolerance = 1e-5 # applies to all instances since it is a static attribute
tio.Subject.absolute_attribute_tolerance = 1e-5

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
    '1.2.826.0.1.3680043.20756',
    '1.2.826.0.1.3680043.29952',
    '1.2.826.0.1.3680043.8362',
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

        # https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/344862
        bad_scans           = ['1.2.826.0.1.3680043.20574', '1.2.826.0.1.3680043.29952']
        for uid in bad_scans:
            self.trainDf.drop(self.trainDf[self.trainDf['StudyInstanceUID'] == uid].index, axis = 0, inplace = True)

        #get the mappings for the data images and segmentations:
        seg_paths = []
        img_paths = []
        UIDs = self.listTrainPatientID()
        for uid in tqdm(UIDs):
            seg_paths.append(os.path.join(self.segPath, str(uid)+".nii"))
            img_paths.append(os.path.join(self.trainImagePath,str(uid)))

        self.trainDf["seg_path"] = seg_paths
        self.trainDf["img_paths"] = img_paths
        self.trainDf.head()

    def bboxFromIndex(self, id, sliceNum):
        box = self.trainBbox.loc[(self.trainBbox.StudyInstanceUID == id) & (self.trainBbox.slice_number == sliceNum), :]
        return list(box.values[0])

    def fracturedBones(self, id):
        fractured_bones = []
        temp = self.trainDf.loc[self.trainDf.StudyInstanceUID == id, ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']]
        temp = list(temp.values[0])  # there is one row per id
        for i in range(len(temp)):
            if temp[i] == 1:
                fractured_bones.append('C' + str(i + 1))
        return fractured_bones

    def listTrainPatientID(self):
        return list(self.trainDf["StudyInstanceUID"])

    def listTestPatientID(self):
        return list(self.testDf["StudyInstanceUID"])

    def loadSliceImageFromId(self, patientID, sliceIndex):
        imgPath = self.trainDf.loc[self.trainDf.StudyInstanceUID == patientID, "img_paths"]
        imgPath = imgPath.iloc[0]
        targetPath = os.path.join(imgPath, str(sliceIndex)+".dcm")
        return loadDicom(targetPath)

    def loadSegmentationsForPatient(self, patientID):
        segmentations = nib.load(os.path.join(self.segPath, patientID+'.nii')).get_fdata()
        segmentations = segmentations[:, ::-1, ::-1]
        segmentations = segmentations.transpose(2, 1, 0)
        return segmentations

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
        downsample = tio.Resample(2)
        cropOrPad = tio.CropOrPad((128,128,200))
        preprocess_spatial = tio.Compose([
            normalize_orientation,
            downsample,
            cropOrPad,
        ])
        sequential = tio.SequentialLabels()
        remapping = [2,3,4,5,6,7,8,9]
        remap_mask = tio.RemoveLabels(remapping,background_label=1)
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
