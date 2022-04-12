import os
import time
import datetime
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

#import network
import segment_network
#import resnet_version
import segment_test_dataset
import segment_utils
import torch.autograd as autograd
from torch.autograd import Variable

import nibabel as nib
import csv
from scipy import ndimage
from skimage import measure
from medpy.io import load,save
import SimpleITK as sitk
from scipy.ndimage import label
import random


def seg_tester_baseline(opt, epoch, segmentor):
    results_path = opt.sample_path + '/' + 'epoch%d' % (epoch)
    if os.path.isdir(results_path) is False:
        os.mkdir(results_path)

    # Define the dataset
    testset = segment_test_dataset.SegmentTestDataset(opt)
    print('The overall number of images equals to %d' % len(testset))

    # Define the dataloader
    dataloader = DataLoader(testset, batch_size = 1, shuffle = False, num_workers = 0, pin_memory = True)
    
    # ----------------------------------------
    #            Testing
    # ----------------------------------------
    # Testing loop
    for batch_idx, (img, liver, file_name) in enumerate(dataloader):
        img = img.cuda()
        liver = liver.cuda()
        file_name = file_name[0]

        # Generator output
        with torch.no_grad():
            # sent to network
            seg_input = img
            seg_output = segmentor(seg_input)

            seg_result = torch.sigmoid(seg_output)
            seg_result[seg_result>opt.threshold]=1
            seg_result[seg_result!=1]=0
        seg_result = seg_result * liver

        segment_utils.save_sample_png(sample_folder = results_path, sample_name = file_name, img=seg_result, pixel_max_cnt = 255)
        print('---------------------' + file_name + ' has been finished----------------------')

    # inpaint results to nii.gz label
    out_folder = opt.sample_path + '/' + 'epoch%d_out_nii' % (epoch)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    segment_utils.pngs_2_niigz(sample_folder = results_path, liver_nii_folder=os.path.join(opt.baseroot_test, 'Liver_nii') , out_folder=out_folder, d08=(opt.dataset==8))
    # evaluate result
    segment_utils.evaluate_nii(dir_out=out_folder, gt_nii_path=os.path.join(opt.baseroot_test, 'Gt_nii'), epoch_name=str(epoch))





def seg_tester_co_ttt(opt, checkpoint_name, lr, iters, generator_path, segmentor_path, reconstructor_path, threshold=0.5):
    # reproduce
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    seed = 66
    print("[ Using Seed : ", seed, " ]")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

    #  set path
    results_path = opt.sample_path + checkpoint_name +  '/' + 'out_png'
    if os.path.isdir(results_path) is False:
        os.mkdir(results_path)


    # Define the dataset
    trainset = segment_test_dataset.SegmentTestDataset(opt)
    print('The overall number of images equals to %d' % len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size = 1, shuffle = False, num_workers = 0, pin_memory = True)

    #-------------------------------------------------------------------------------------------------------------------------
    # Build networks
    generator = segment_utils.create_Unet(opt, n_type='U64')
    reconstructor = segment_utils.create_Unet(opt, n_type='U64', in_channels=2)
    segmentor = segment_utils.create_Unet(opt, n_type='U64')


    # To device
    generator = generator.cuda()
    segmentor = segmentor.cuda()
    reconstructor = reconstructor.cuda()


    # load parameters
    generator.load_state_dict(torch.load(generator_path))
    segmentor.load_state_dict(torch.load(segmentor_path))
    reconstructor.load_state_dict(torch.load(reconstructor_path))


    # Optimizers
    parameterg = list(generator.parameters())
    optimizer_g = torch.optim.Adam(parameterg, lr = lr,  betas=(0.9, 0.99))
    parameters = list(segmentor.parameters()) 
    optimizer_s = torch.optim.Adam(parameters, lr = lr,  betas=(0.9, 0.99))

    #-------------------------------------------------------------------------------------------------------------------------
    
    # ----------------------------------------
    #            Testing
    # ----------------------------------------
    # Testing loop
    for batch_idx, (img, liver, lesion, file_name) in enumerate(dataloader):
        img = img.cuda()
        lesion = lesion.cuda()
        liver = liver.cuda()
        file_name = file_name[0]

        # load parameters
        generator.load_state_dict(torch.load(generator_path))
        segmentor.load_state_dict(torch.load(segmentor_path))
        reconstructor.load_state_dict(torch.load(reconstructor_path))
        generator.eval()
        segmentor.eval()
        reconstructor.eval()
        # get original images
        with torch.no_grad():
            # generator
            gen_output = torch.sigmoid(generator(img))
            # segmentor
            seg_output = segmentor(img)
            
            #-----------------------------sign--------------------------------------
            sign_img = torch.sign(img - gen_output)*liver[:,0:1,:,:]

            # reconstructor 
            re_output = torch.sigmoid(reconstructor(torch.cat((gen_output, torch.sigmoid(seg_output)), 1)))

            seg_result = torch.sigmoid(seg_output)
            seg_result[seg_result>threshold]=1
            seg_result[seg_result!=1]=0

        seg_result_ori = seg_result * liver
        gen_output_ori = img * (1 - liver) + gen_output * liver
        re_output_ori = img * (1 - liver) + re_output * liver


        generator.eval()
        segmentor.eval()
        reconstructor.eval()
        # update segmentor
        segmentor.train()
        for n_iter in range(iters):            
            optimizer_s.zero_grad()
            # generator
            gen_output = torch.sigmoid(generator(img))
            # segmentor
            seg_output_tumor = segmentor(img)
            # reconstructor
            re_output = torch.sigmoid(reconstructor(torch.cat((gen_output.detach(), torch.sigmoid(seg_output_tumor)), 1)))
            # refine reoutput
            re_output_liver = img * (1 - liver) + re_output * liver

            # calculate reconstruction loss
            loss_criterion_r = torch.nn.L1Loss()
            loss_r = loss_criterion_r(re_output_liver, img)
                
            loss_total_s = loss_r
            loss_total_s.backward()
            optimizer_s.step()
            print("\r[Batch %d/%d] [R Loss: %.5f] " % ((batch_idx+1), len(dataloader), loss_total_s.item()))


        # start test
        generator.eval()
        segmentor.eval()
        reconstructor.eval()

        with torch.no_grad():
            # generator
            gen_output = torch.sigmoid(generator(img))
            # segmentor
            seg_output = segmentor(img)

            # reconstructor 
            re_output = torch.sigmoid(reconstructor(torch.cat((gen_output, torch.sigmoid(seg_output)), 1)))

            seg_result = torch.sigmoid(seg_output)
            seg_result[seg_result>threshold]=1
            seg_result[seg_result!=1]=0

        seg_result = seg_result * liver
        re_output = img * (1 - liver) + re_output * liver

        segment_utils.save_sample_png(sample_folder = results_path, sample_name = file_name, img=seg_result, pixel_max_cnt = 255)
        print('---------------------' + file_name + ' has been finished----------------------')


    # inpaint results to nii.gz label
    out_folder = opt.sample_path + checkpoint_name + '/' + 'out_nii'
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    segment_utils.pngs_2_niigz(sample_folder = results_path, liver_nii_folder=os.path.join(opt.baseroot_test, 'Liver_nii') , out_folder=out_folder, d08=(opt.dataset==8))
    # evaluate result
    segment_utils.evaluate_nii(dir_out=out_folder, gt_nii_path=os.path.join(opt.baseroot_test, 'Gt_nii'), epoch_name='12')

