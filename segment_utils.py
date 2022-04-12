import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision as tv
import torchvision.models as models
from segment_network import *
import glob
from helpers.calc_metric import (dice,
                                 detect_lesions,
                                 compute_segmentation_scores,
                                 compute_tumor_burden,
                                 LARGE)
from helpers.utils import time_elapsed
import nibabel as nib
from scipy.ndimage.measurements import label as label_connected_components
import gc
import pandas as pd
import random

import csv
from scipy import ndimage
from skimage import measure
from medpy.io import load,save
import SimpleITK as sitk
from scipy.ndimage import label

# ----------------------------------------
#                 Network
# ----------------------------------------
def create_Unet(opt, in_channels = 1):
    unet_model = UNet(in_channels=in_channels)
    weights_init(unet_model, init_type = opt.init_type, init_gain = opt.init_gain)
    return unet_model



def load_dict(process_net, pretrained_net):
    # Get the dict from pre-trained network
    pretrained_dict = pretrained_net.state_dict()
    # Get the dict from processing network
    process_dict = process_net.state_dict()
    # Delete the extra keys of pretrained_dict that do not belong to process_dict
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in process_dict}
    # Update process_dict using pretrained_dict
    process_dict.update(pretrained_dict)
    # Load the updated dict to processing network
    process_net.load_state_dict(process_dict)
    return process_net


def weights_init(net, init_type='kaiming', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)       -- network to be initialized
        init_type (str)     -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_var (float)    -- scaling factor for normal, xavier and orthogonal.
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            init.normal_(m.weight, 0, 0.01)
            init.constant_(m.bias, 0)

    # Apply the initialization function <init_func>
    net.apply(init_func)
    
# ---------------------------------------
    
# ----------------------------------------
#    Validation 
# ----------------------------------------
def save_sample_png(sample_folder, sample_name, img, pixel_max_cnt = 255):
    volume_num = sample_name.split('/')[0]
    volume_path = os.path.join(sample_folder, volume_num)
    if not os.path.exists(volume_path):
        os.makedirs(volume_path)
    img = img * 255
    img_copy = img.clone().data[0,0].cpu().numpy()
    img_copy = np.clip(img_copy, 0, pixel_max_cnt)
    img_copy = img_copy.astype(np.uint8)
    #--------------threshold------------------------------------------------
    ret2,img_copy = cv2.threshold(img_copy,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #cv2.imshow('result', th2)
    #cv2.waitKey(0)
    #------------------------------------------------------------------------
    # Save to certain path
    save_img_path = os.path.join(sample_folder, sample_name)
    cv2.imwrite(save_img_path, img_copy)


def pngs_2_niigz(sample_folder, liver_nii_folder, out_folder, d08=True):
    label_files = os.listdir(liver_nii_folder)
    for label_file in label_files:
        # load original nii
        ori_path= os.path.join(liver_nii_folder, label_file)
        ori_nii = sitk.ReadImage(ori_path , sitk.sitkUInt8)
        ori_data = sitk.GetArrayFromImage(ori_nii)

        #-----------------------------------------------------
        if d08:
            volume_num = ((label_file.split('_')[1]).split('.'))[0]
        else:
            volume_num = ((label_file.split('-')[2]).split('.'))[0]

        lesion_folder = os.path.join(sample_folder, str(int(volume_num)))

        if os.path.exists(lesion_folder):
            lesion_files = os.listdir(lesion_folder)
            for lesion_file in lesion_files:
                lesion_file_slice = int(lesion_file.split('.')[0])
                lesion_file_data = cv2.imread(os.path.join(lesion_folder, lesion_file), -1)
                ori_data_slice = ori_data[lesion_file_slice, :,:]
                ori_data_slice[lesion_file_data==255]=2
                ori_data[lesion_file_slice, :,:] = ori_data_slice

        #save nii
        out_path = os.path.join(out_folder, label_file)
        img_new = sitk.GetImageFromArray(ori_data)
        img_new.CopyInformation(ori_nii)
        sitk.WriteImage(img_new, out_path)
        print(label_file)


def evaluate_nii(dir_out, gt_nii_path, epoch_name):
    # Segmentation metrics and their default values for when there are no detected
    # objects on which to evaluate them.
    #
    # Surface distance (and volume difference) metrics between two masks are
    # meaningless when any one of the masks is empty. Assign maximum (infinite)
    # penalty. The average score for these metrics, over all objects, will thus
    # also not be finite as it also loses meaning.
    segmentation_metrics = {'dice': 0,
                            'jaccard': 0,
                            'voe': 1,
                            'rvd': LARGE,
                            'assd': LARGE,
                            'rmsd': LARGE,
                            'msd': LARGE}

    # Initialize results dictionaries
    lesion_detection_stats = {0:   {'TP': 0, 'FP': 0, 'FN': 0},
                            0.5: {'TP': 0, 'FP': 0, 'FN': 0}}
    lesion_segmentation_scores = {}
    liver_segmentation_scores = {}
    dice_per_case = {'lesion': [], 'liver': []}
    dice_global_x = {'lesion': {'I': 0, 'S': 0},
                    'liver':  {'I': 0, 'S': 0}} # 2*I/S
    tumor_burden_list = []

    # Iterate over all volumes in the reference list.

    reference_volume_list = sorted(glob.glob(gt_nii_path + '/*.nii.gz'))

    for reference_volume_fn in reference_volume_list:
        print("Starting with volume {}".format(reference_volume_fn))
        submission_volume_path = os.path.join(dir_out, os.path.basename(reference_volume_fn))
        t = time_elapsed()

        # Load reference and submission volumes with Nibabel.
        reference_volume = nib.load(reference_volume_fn)
        submission_volume = nib.load(submission_volume_path)


        # Get the current voxel spacing.
        voxel_spacing = submission_volume.header.get_zooms()[:3]

        # Get Numpy data and compress to int8.
        reference_volume = (reference_volume.get_data()).astype(np.int8)
        submission_volume = (submission_volume.get_data()).astype(np.int8)
        
        # Ensure that the shapes of the masks match.
        if submission_volume.shape!=reference_volume.shape:
            raise AttributeError("Shapes do not match! Prediction mask {}, "
                                "ground truth mask {}"
                                "".format(submission_volume.shape,
                                        reference_volume.shape))
        #print("Done loading files ({:.2f} seconds)".format(t()))
        
        # Create lesion and liver masks with labeled connected components.
        # (Assuming there is always exactly one liver - one connected comp.)
        pred_mask_lesion, num_predicted = label_connected_components( \
                                            submission_volume==2, output=np.int16)
        true_mask_lesion, num_reference = label_connected_components( \
                                            reference_volume==2, output=np.int16)
        pred_mask_liver = submission_volume>=1
        true_mask_liver = reference_volume>=1
        liver_prediction_exists = np.any(submission_volume==1)
        #print("Done finding connected components ({:.2f} seconds)".format(t()))
        
        # Identify detected lesions.
        # Retain detected_mask_lesion for overlap > 0.5
        for overlap in [0, 0.5]:
            detection_result = detect_lesions( \
                                                prediction_mask=pred_mask_lesion,
                                                reference_mask=true_mask_lesion,
                                                min_overlap=overlap)
            detected_mask_lesion, mod_ref_mask, num_detected = detection_result[0:3]
            
            # Count true/false positive and false negative detections.
            lesion_detection_stats[overlap]['TP']+=num_detected
            lesion_detection_stats[overlap]['FP']+=num_predicted-num_detected
            lesion_detection_stats[overlap]['FN']+=num_reference-num_detected
        #print("Done identifying detected lesions ({:.2f} seconds)".format(t()))
        
        # Compute segmentation scores for DETECTED lesions.
        if num_detected>0:
            lesion_scores = compute_segmentation_scores( \
                                            prediction_mask=detected_mask_lesion,
                                            reference_mask=mod_ref_mask,
                                            voxel_spacing=voxel_spacing)
            for metric in segmentation_metrics:
                if metric not in lesion_segmentation_scores:
                    lesion_segmentation_scores[metric] = []
                lesion_segmentation_scores[metric].extend(lesion_scores[metric])
            #print("Done computing lesion scores ({:.2f} seconds)".format(t()))
        else:
            a=1
            #print("No lesions detected, skipping lesion score evaluation")
        
        # Compute liver segmentation scores. 
        if liver_prediction_exists:
            liver_scores = compute_segmentation_scores( \
                                            prediction_mask=pred_mask_liver,
                                            reference_mask=true_mask_liver,
                                            voxel_spacing=voxel_spacing)
            for metric in segmentation_metrics:
                if metric not in liver_segmentation_scores:
                    liver_segmentation_scores[metric] = []
                liver_segmentation_scores[metric].extend(liver_scores[metric])
            #print("Done computing liver scores ({:.2f} seconds)".format(t()))
        else:
            # No liver label. Record default score values (zeros, inf).
            # NOTE: This will make some metrics evaluate to inf over the entire
            # dataset.
            for metric in segmentation_metrics:
                if metric not in liver_segmentation_scores:
                    liver_segmentation_scores[metric] = []
                liver_segmentation_scores[metric].append(\
                                                    segmentation_metrics[metric])
            #print("No liver label provided, skipping liver score evaluation")
            
        # Compute per-case (per patient volume) dice.
        if not np.any(pred_mask_lesion) and not np.any(true_mask_lesion):
            dice_per_case['lesion'].append(1.)
        else:
            dice_per_case['lesion'].append(dice(pred_mask_lesion,
                                                true_mask_lesion))
        if liver_prediction_exists:
            dice_per_case['liver'].append(dice(pred_mask_liver,
                                            true_mask_liver))
        else:
            dice_per_case['liver'].append(0)
        
        # Accumulate stats for global (dataset-wide) dice score.
        dice_global_x['lesion']['I'] += np.count_nonzero( \
            np.logical_and(pred_mask_lesion, true_mask_lesion))
        dice_global_x['lesion']['S'] += np.count_nonzero(pred_mask_lesion) + \
                                        np.count_nonzero(true_mask_lesion)
        if liver_prediction_exists:
            dice_global_x['liver']['I'] += np.count_nonzero( \
                np.logical_and(pred_mask_liver, true_mask_liver))
            dice_global_x['liver']['S'] += np.count_nonzero(pred_mask_liver) + \
                                        np.count_nonzero(true_mask_liver)
        else:
            # NOTE: This value should never be zero.
            dice_global_x['liver']['S'] += np.count_nonzero(true_mask_liver)
            
            
        #print("Done computing additional dice scores ({:.2f} seconds)""".format(t()))
            
        # Compute tumor burden.
        tumor_burden = compute_tumor_burden(prediction_mask=submission_volume,
                                            reference_mask=reference_volume)
        tumor_burden_list.append(tumor_burden)
        #print("Done computing tumor burden diff ({:.2f} seconds)".format(t()))
        
        gc.collect()
            
            
    # Compute lesion detection metrics.
    _det = {}
    for overlap in [0, 0.5]:
        TP = lesion_detection_stats[overlap]['TP']
        FP = lesion_detection_stats[overlap]['FP']
        FN = lesion_detection_stats[overlap]['FN']
        precision = float(TP)/(TP+FP) if TP+FP else 0
        recall = float(TP)/(TP+FN) if TP+FN else 0
        _det[overlap] = {'p': precision, 'r': recall}
    lesion_detection_metrics = {'precision': _det[0.5]['p'],
                                'recall': _det[0.5]['r'],
                                'precision_greater_zero': _det[0]['p'],
                                'recall_greater_zero': _det[0]['r']}

    # Compute lesion segmentation metrics.
    lesion_segmentation_metrics = {}
    for m in lesion_segmentation_scores:
        lesion_segmentation_metrics[m] = np.mean(lesion_segmentation_scores[m])
    if len(lesion_segmentation_scores)==0:
        # Nothing detected - set default values.
        lesion_segmentation_metrics.update(segmentation_metrics)
    lesion_segmentation_metrics['dice_per_case'] = np.mean(dice_per_case['lesion'])
    dice_global = 2.*dice_global_x['lesion']['I']/dice_global_x['lesion']['S']
    lesion_segmentation_metrics['dice_global'] = dice_global
        
    # Compute liver segmentation metrics.
    liver_segmentation_metrics = {}
    for m in liver_segmentation_scores:
        liver_segmentation_metrics[m] = np.mean(liver_segmentation_scores[m])
    if len(liver_segmentation_scores)==0:
        # Nothing detected - set default values.
        liver_segmentation_metrics.update(segmentation_metrics)
    liver_segmentation_metrics['dice_per_case'] = np.mean(dice_per_case['liver'])
    dice_global = 2.*dice_global_x['liver']['I']/dice_global_x['liver']['S']
    liver_segmentation_metrics['dice_global'] = dice_global

    # Compute tumor burden.
    tumor_burden_rmse = np.sqrt(np.mean(np.square(tumor_burden_list)))
    tumor_burden_max = np.max(tumor_burden_list)


    # Print results to stdout.
    print("Computed LESION DETECTION metrics:")
    for metric, value in lesion_detection_metrics.items():
        print("{}: {:.3f}".format(metric, float(value)))
    print("Computed LESION SEGMENTATION metrics (for detected lesions):")
    for metric, value in lesion_segmentation_metrics.items():
        print("{}: {:.3f}".format(metric, float(value)))
    print("Computed LIVER SEGMENTATION metrics:")
    #for metric, value in liver_segmentation_metrics.items():
        #print("{}: {:.3f}".format(metric, float(value)))
    print("Computed TUMOR BURDEN: \n"
        "rmse: {:.3f}\nmax: {:.3f}".format(tumor_burden_rmse, tumor_burden_max))

    # Write metrics to file.
    output_filename = os.path.join(dir_out, str(epoch_name)+'_scores.txt')
    output_file = open(output_filename, 'w')
    for metric, value in lesion_detection_metrics.items():
        output_file.write("lesion_{}: {:.3f}\n".format(metric, float(value)))
    for metric, value in lesion_segmentation_metrics.items():
        output_file.write("lesion_{}: {:.3f}\n".format(metric, float(value)))
    #for metric, value in liver_segmentation_metrics.items():
        #output_file.write("liver_{}: {:.3f}\n".format(metric, float(value)))

    #Tumorburden
    output_file.write("RMSE_Tumorburden: {:.3f}\n".format(tumor_burden_rmse))
    output_file.write("MAXERROR_Tumorburden: {:.3f}\n".format(tumor_burden_max))

    output_file.close()



    # 将评价指标写入到exel中
    liver_data = pd.DataFrame(lesion_segmentation_scores)

    liver_statistics = pd.DataFrame(index=['mean', 'std', 'min', 'max'], columns=list(liver_data.columns))
    liver_statistics.loc['mean'] = liver_data.mean()
    liver_statistics.loc['std'] = liver_data.std()
    liver_statistics.loc['min'] = liver_data.min()
    liver_statistics.loc['max'] = liver_data.max()

    writer = pd.ExcelWriter(os.path.join(dir_out, str(epoch_name)+'_result.xlsx'))
    liver_data.to_excel(writer, 'liver')
    liver_statistics.to_excel(writer, 'liver_statistics')
    writer.save()

    lesion_dpc = lesion_segmentation_metrics['dice_per_case']
    return lesion_dpc




