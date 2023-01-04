from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import cv2
import time
from PIL import Image
import os
import torch

class CustomDataset(Dataset):
    def __init__(self, mode, transform):
        self.image_path = "../data/frames"
        self.transform = transform
        self.mode = mode

        self.images_df = pd.read_pickle("../data/images.pkl")
        self.train_df = pd.read_pickle("../data/train.pkl")
        self.points3D_df = pd.read_pickle("../data/points3D.pkl")
        self.point_desc_df = pd.read_pickle("../data/point_desc.pkl")

        filenames = np.array(self.images_df["NAME"].to_list())
        self.gt_t = np.vstack((self.images_df["TX"].values, self.images_df["TY"].values, self.images_df["TZ"].values)).transpose((1,0)) #(N,3)
        self.gt_R = np.vstack((self.images_df["QX"].values, self.images_df["QY"].values, self.images_df["QZ"].values,self.images_df["QW"].values)).transpose((1,0))#(N,4)
        

        self.filenames = []
        self.train_t = []
        self.train_R = []
        
        for i, fn in enumerate(filenames):
            if self.mode in fn:
                self.filenames.append(fn)
                self.train_t.append(self.gt_t[i])
                self.train_R.append(self.gt_R[i])




        self.num = self.filenames.__len__()
        print("Number of "+self.mode, self.num)

    def __getitem__(self, index):
        rimg = Image.open(os.path.join(self.image_path,self.filenames[index]))

        t = self.train_t[index]
        R = self.train_R[index]

        return self.transform(rimg), torch.Tensor(t), torch.Tensor(R)

    def __len__(self):

        return self.filenames.__len__()