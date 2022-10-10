import itertools
import os
import time
import datetime
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.nn import functional as F

#import network
import segment_network
#import resnet_version
import segment_train_dataset
import segment_test_dataset
import segment_utils
import torch.autograd as autograd
from torch.autograd import Variable
import segment_tester
import random
from torch import optim
import math


# Save the model if pre_train == True
def save_model(net, epoch, opt, name, save_folder):
    model_name = name + '_epoch%d.pth' % (epoch+1)
    model_name = os.path.join(save_folder, model_name)
    torch.save(net.state_dict(), model_name)
    print('The trained model is successfully saved at epoch %d' % (epoch))


# baseline: directly train a segmentation network
def seg_trainer_baseline(opt):
    # ----------------------------------------
    #      Initialize training parameters
    # ----------------------------------------    
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

    # configurations
    save_folder = opt.save_path
    sample_folder = opt.sample_path
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if not os.path.exists(sample_folder):
        os.makedirs(sample_folder)
    
    # Build networks
    segmentor = segment_utils.create_Unet(opt)
    # To device
    segmentor = segmentor.cuda()
    # Optimizers
    optimizer_s = torch.optim.Adam(segmentor.parameters(), lr = opt.lr, betas=(0.9, 0.99))

    # ----------------------------------------
    #       Initialize training dataset
    # ----------------------------------------
    # Define the dataset
    trainset = segment_train_dataset.SegmentTrainDataset(opt)
    print('The overall number of images equals to %d' % len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    
    # ----------------------------------------
    #            Training
    # ----------------------------------------
    # Initialize start time
    prev_time = time.time()

    # Training loop
    for epoch in range(opt.epochs):
        print("Start epoch ", epoch+1, "!")
        for batch_idx, (img, synthesis_img, synthesis_mask, liver_mask) in enumerate(dataloader):
            # sent images to cuda
            img = img.cuda()
            synthesis_img = synthesis_img.cuda()
            synthesis_mask = synthesis_mask.cuda()
            
            # sent to network
            seg_input = synthesis_img
            seg_output = segmentor(seg_input)

            # loss and optimizer
            optimizer_s.zero_grad()
            pos_weight = (opt.loss_weight*torch.ones([1])).cuda()
            loss_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loss_S = loss_criterion(seg_output, synthesis_mask)

            loss_S.backward()
            optimizer_s.step()
        
            # Determine approximate time left
            batches_done = epoch * len(dataloader) + batch_idx
            batches_left = opt.epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()
            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [S Loss: %.5f] " % ((epoch + 1), opt.epochs, (batch_idx+1), len(dataloader), loss_S.item()))


        # Save the model
        save_model(segmentor, epoch , opt, save_folder)




# our method
def seg_trainer_ttt(opt):
    # ----------------------------------------
    #      Initialize training parameters
    # ----------------------------------------
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


    # configurations
    save_folder = opt.save_path
    sample_folder = opt.sample_path
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if not os.path.exists(sample_folder):
        os.makedirs(sample_folder)
    

    # Build networks
    generator = segment_utils.create_Unet(opt)
    reconstructor = segment_utils.create_Unet(opt, in_channels=2)
    segmentor = segment_utils.create_Unet(opt)


    # To device
    generator = generator.cuda()
    segmentor = segmentor.cuda()
    reconstructor = reconstructor.cuda()
 

    # Optimizers
    parameterg = list(generator.parameters())
    optimizer_g = torch.optim.Adam(parameterg, lr = opt.lr, betas=(0.9, 0.99))
    parameters = list(segmentor.parameters()) + list(reconstructor.parameters())
    optimizer_s = torch.optim.Adam(parameters, lr = opt.lr, betas=(0.9, 0.99))
    
    # ----------------------------------------
    #       Initialize training dataset
    # ----------------------------------------

    # Define the dataset
    trainset = segment_train_dataset.SegmentTrainDataset(opt)
    print('The overall number of images equals to %d' % len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    
    # ----------------------------------------
    #            Training and Testing
    # ----------------------------------------

    # Initialize start time
    prev_time = time.time()

    # Training loop
    for epoch in range(opt.epochs):
        print("Start epoch ", epoch+1, "!")

        for batch_idx, (img, synthesis_img, synthesis_mask, liver_mask) in enumerate(dataloader):
            
            teacher_forcing_ratio = 1-1.0/2.0*(epoch + (batch_idx+1.0)/len(dataloader))
            if teacher_forcing_ratio<0:
                teacher_forcing_ratio = 0
            print("teacher_forcing_ratio ", str(teacher_forcing_ratio), "!")

            # sent images to cuda
            img = img.cuda()
            liver_mask = liver_mask.cuda()
            synthesis_img = synthesis_img.cuda()
            synthesis_mask = synthesis_mask.cuda()

            # step 1: update segmentor and reconstructor
            optimizer_s.zero_grad()
            # segmentor
            seg_output_tumor = segmentor(synthesis_img)
            seg_output_healthy = segmentor(img)
            # generator
            gen_output = torch.sigmoid(generator(synthesis_img))

            #---------------------------------------------------------------------------------------------------------
            # different input to reconstructor
            re_input = teacher_forcing_ratio*img + (1-teacher_forcing_ratio)*gen_output.detach()
            re_output = torch.sigmoid(reconstructor(torch.cat((re_input, torch.sigmoid(seg_output_tumor)), 1)))
            # refine reoutput
            re_output_liver = img * (1 - liver_mask) + re_output * liver_mask
           
            # calculate reconstruction loss            
            loss_criterion_L1 = torch.nn.L1Loss()
            loss_r = loss_criterion_L1(re_output_liver, synthesis_img) 
            # calculate segmentation loss
            pos_weight = (opt.loss_weight*torch.ones([1])).cuda()
            loss_criterion_s = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loss_s_tumor = loss_criterion_s(seg_output_tumor, synthesis_mask) 
            loss_s_healthy = loss_criterion_s(seg_output_healthy, 0*synthesis_mask)
           
            # total loss
            w_r = 1
            w_st = 2
            w_sh = 1
            loss_total_s = w_r*loss_r + w_st*loss_s_tumor + w_sh*loss_s_healthy
            loss_total_s.backward()
            optimizer_s.step()


            # step 2: update generator
            optimizer_g.zero_grad()
            # generator
            gen_output = torch.sigmoid(generator(synthesis_img))
            # refine geoutput
            gen_output_liver = img * (1 - liver_mask) + gen_output * liver_mask
            # calculate generation loss
            loss_g = loss_criterion_L1(gen_output_liver, img) 
            # calculate segmentation loss
            seg_output_gen = segmentor(gen_output_liver)
            loss_gen_s = loss_criterion_s(seg_output_gen, 0*synthesis_mask)
            # total loss
            w_g = 1
            w_gs = 0.1
            loss_total_g = w_g*loss_g + w_gs*loss_gen_s
            loss_total_g.backward()
            optimizer_g.step()


            # Determine approximate time left
            batches_done = epoch * len(dataloader) + batch_idx
            batches_left = opt.epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()
            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [Ratio:%d/%d/%d/%d/%d][loss_r: %.5f] [loss_s_tumor: %.5f] [loss_s_healthy: %.5f] [loss_g: %.5f] [loss_gen_s: %.5f]" % 
                ((epoch + 1), opt.epochs, (batch_idx+1), len(dataloader), w_r, w_st, w_sh, w_g, w_gs, loss_r.item(), loss_s_tumor.item(), loss_s_healthy.item(), loss_g.item(), loss_gen_s.item()))

         
        # Save the model
        save_model(generator, epoch , opt, 'generator', save_folder)
        save_model(segmentor, epoch , opt, 'segmentor', save_folder)
        save_model(reconstructor, epoch , opt, 'reconstructor', save_folder)


        





