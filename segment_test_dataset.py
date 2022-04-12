import os
import cv2
import numpy as np
import torch
import random
import math
from PIL import Image, ImageDraw
from skimage.restoration import denoise_nl_means, estimate_sigma
from torchvision import transforms
from torch.utils.data import Dataset
import pandas as pd
from medpy.filter.smoothing import anisotropic_diffusion
from skimage.filters import meijering, sato, frangi, hessian


def windowing(im, win = [-200, 250]):
    # scale intensity from win[0]~win[1] to float numbers in 0~255
    im1 = im.astype(float)
    im1 -= win[0]
    im1 /= win[1] - win[0]
    im1[im1 > 1] = 1
    im1[im1 < 0] = 0
    #im1 *= 255
    return im1


class SegmentTestDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        
        self.img_root = os.path.join(opt.baseroot_test, 'Image') 
        self.liver_root = os.path.join(opt.baseroot_test, 'Liver') 
        self.test_config_file = os.path.join(opt.baseroot_test, 'Config/lits_test.xlsx') 

        excelData = pd.read_excel(self.test_config_file)
        length = excelData.shape[0] 
        
        self.paths = []
        self.paths_attr = {}
        for i in range(length): 
            file_name =  excelData.iloc[i][0]
            self.paths.append(file_name)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        # load liver
        liver_path = os.path.join(self.liver_root, self.paths[index]) 
        liver_array = np.asarray(Image.open(liver_path))
        liver_arrays = [liver_array.astype(float), liver_array.astype(float), liver_array.astype(float)]
        liver_arrays = cv2.merge(liver_arrays)
        liver = liver_arrays.astype(np.float32, copy=False)

        # load image
        img_path = os.path.join(self.img_root, self.paths[index]) 
        img_array = np.asarray(Image.open(img_path))
        img_array = img_array.astype(np.float32, copy=False)-32768 
        img_array = windowing(img_array, win = [-200, 250])*liver_array
        img_array_de = anisotropic_diffusion(img_array.astype(float), niter=20, kappa=20, gamma=0.2, voxelspacing=None, option=3)
        ims = [img_array_de, img_array_de, img_array_de]
        # combine images
        ims = [im.astype(float) for im in ims]
        img = cv2.merge(ims)
        #-------------------------------------------------------------------------

        img = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1).contiguous()
        liver = torch.from_numpy(liver.astype(np.float32)).permute(2, 0, 1).contiguous()

        return img[0:1], liver, self.paths[index]
