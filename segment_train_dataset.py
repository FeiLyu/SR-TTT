import os
import cv2
import numpy as np
import torch
import random
import math
from PIL import Image, ImageDraw

from torchvision import transforms
from torch.utils.data import Dataset
import pandas as pd

import segment_utils
from medpy.filter.smoothing import anisotropic_diffusion
from skimage.segmentation import slic, mark_boundaries

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


def windowing(im, win = [-200, 250]):
    # scale intensity from win[0]~win[1] to float numbers in 0~255
    im1 = im.astype(float)
    im1 -= win[0]
    im1 /= win[1] - win[0]
    im1[im1 > 1] = 1
    im1[im1 < 0] = 0
    #im1 *= 255
    return im1

def elastic_transform(image, alpha, sigma):
    assert len(image.shape)==2
    shape = image.shape
    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))
    return map_coordinates(image, indices, order=1).reshape(shape)

class SegmentTrainDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        
        self.img_root = os.path.join(opt.baseroot, 'Image') 
        self.liver_root = os.path.join(opt.baseroot, 'Liver') 
        self.healthy_config_file = os.path.join(opt.baseroot, 'Config/lits_train_healthy.xlsx') 

        self.a1 = opt.a1
        self.b1 = opt.b1
        self.a2 = opt.a2
        self.b2 = opt.b2

        # read images
        excelData = pd.read_excel(self.healthy_config_file)
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

        #--------------------------------------------------------------------------
        # generate synthesis mask
        ROI = np.uint8(liver[:,:,0])
        if np.sum(ROI)>200:
            n_segments = np.random.randint(low=2, high=45)
            # slice segmentation
            slic_sp = slic(img, n_segments=n_segments, compactness=10, start_label=1, mask = ROI)
            # check sp num and remove 0
            index_slic_all = np.unique(slic_sp)
            index_slic_all = index_slic_all[index_slic_all != 0]
            num_slic = len(np.unique(index_slic_all))
            # when sp num > 1
            if num_slic>1:
                index_slic_selected = random.choice(index_slic_all)
            
                # get mask of selected superpixel
                mask_slic = np.zeros_like(ROI)
                mask_slic[slic_sp==index_slic_selected]=1

                contours,hierarchy = cv2.findContours(mask_slic.astype(np.uint8), 1, 2)
                cnt = contours[0]
                if cnt.shape[0]>5:
                    ellipse = cv2.fitEllipse(cnt)
                    ellipse_mask = (np.zeros_like(ROI)).astype(np.uint8)
                    cv2.ellipse(ellipse_mask, ellipse, 1, -1)
                    ellipse_mask = ellipse_mask * ROI

                    alpha = np.random.randint(low=20, high=200)
                    sigma = np.random.randint(low=4, high=10)
                    synthesis_mask = elastic_transform(ellipse_mask, alpha, sigma)
                else:
                    synthesis_mask = ROI
            else:
                synthesis_mask = ROI
        else:
            synthesis_mask = ROI
        
        synthesis_mask = synthesis_mask * ROI


        # intensity generation
        mean_value = 255*np.sum(img_array*synthesis_mask)/(np.sum(synthesis_mask)+1e-6)
        mu = -100 + 150*np.random.beta(a=self.a1, b=self.b1, size=1)[0]
        sigma = 5 + 60*np.random.beta(a=self.a2, b=self.b2, size=1)[0]
        noise = np.random.normal(mean_value+mu, sigma, (512, 512))
        noise = np.clip(noise, 0, 255)        
        noise = noise/255.0


        # synthesis image
        slic_image_array = img_array*(1-synthesis_mask) + noise*synthesis_mask
        slic_image_array_de = anisotropic_diffusion(slic_image_array.astype(float), niter=20, kappa=20, gamma=0.2, voxelspacing=None, option=3)
        slice_ims = [slic_image_array_de, slic_image_array_de, slic_image_array_de]
        # combine images
        slic_ims = [im.astype(float) for im in slice_ims]
        synthesis_img = cv2.merge(slic_ims)


        #--------------------------------------------------------------------------
        img = torch.from_numpy(img.astype(np.float32) ).permute(2, 0, 1).contiguous()
        synthesis_img = torch.from_numpy(synthesis_img.astype(np.float32) ).permute(2, 0, 1).contiguous()
        liver_mask = torch.from_numpy(liver.astype(np.float32)).permute(2, 0, 1).contiguous()
        synthesis_mask = torch.from_numpy((np.expand_dims(synthesis_mask, axis=2)).astype(np.float32)).permute(2, 0, 1).contiguous()

        return img[0:1], synthesis_img[0:1], synthesis_mask, liver_mask[0:1]
