
from PIL import Image

import pandas as pd
import pydicom
import nibabel as nib
import numpy as np
import json
import cv2
import os
from tqdm import tqdm

with open('config.json', 'r') as f:
    paths = json.load(f)

RSNA_2022_PATH    = paths["RSNA_2022_PATH"]
TRAIN_IMAGES_PATH = f'{RSNA_2022_PATH}/train_images'
TEST_IMAGES_PATH  = f'{RSNA_2022_PATH}/test_images'

class KaggleDataLoader:

    def __init__(self):
        # Load metadata

        self.trainImagePath = TRAIN_IMAGES_PATH
        self.segPath        = os.path.join(TRAIN_IMAGES_PATH,          "segmentations")
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
        box = self.trainBbox.loc[(self.trainBbox.StudyInstanceUID == id & self.trainBbox.slice_number == sliceNum), :]
        return list(box.values[0])

    def fracturedBones(self, patientID):
        fractured_bones = []
        temp = self.trainDf.loc[self.trainDf.StudyInstanceUID == patientID, ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']]
        temp = list(temp.values[0])  # there is one row per id
        for i in range(len(temp)):
            if temp[i] == 1:
                fractured_bones.append('C' + str(i + 1))
        return fractured_bones

    def listTrainPatientID(self):
        return list(self.trainDf["StudyInstanceUID"])

    def listTestPatientID(self):
        return list(self.testDf["StudyInstanceUID"])


    def loadDicom(self, path):
        img = pydicom.dcmread(path)
        img.PhotometricInterpretation = 'YBR_FULL'
        data = img.pixel_array
        data = data - np.min(data)
        if np.max(data) != 0:
            data = data / np.max(data)
        data = (data * 255).astype(np.uint8)
        return cv2.cvtColor(data, cv2.COLOR_GRAY2RGB), img

    def loadSliceImageFromId(self, patientID, sliceIndex):
        imgPath = self.trainDf.loc[self.trainDf.StudyInstanceUID == patientID, "img_paths"]
        targetPath = os.path.join(imgPath, str(sliceIndex), ".dcm")
        return self.loadDicom(targetPath)

    def loadSegmentationsForPatient(self, patientID):
        segmentations = nib.load(os.path.join(self.segPath, patientID, '.nii')).get_fdata()
        segmentations = segmentations[:, ::-1, ::-1]
        segmentations = segmentations.transpose(2, 1, 0)
        return segmentations

    ## Dataset generator functions
    def loadDatasetAsClassifier(self):
        """
        prepare full dataset for training
        """
    def loadDatasetAsDetector(self):
        """
        prepare full dataset for training
        """
    def loadDatasetAsSegmentor(self):
        """
        prepare full dataset for training
        """
    def loadDatasetAsMaskRCNN(self):
        """
        prepare full dataset for training
        """