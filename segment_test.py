import argparse
import os
import torch

if __name__ == "__main__":

    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # General parameters
    parser.add_argument('--save_path', type = str, default = './models/', help = 'saving path that is a folder')
    parser.add_argument('--sample_path', type = str, default = './samples/', help = 'training samples path that is a folder')
    
    # Network parameters
    parser.add_argument('--init_type', type = str, default = 'kaiming', help = 'the initialization type')
    parser.add_argument('--init_gain', type = float, default = 0.2, help = 'the initialization gain')


    # Dataset path
    parser.add_argument('--baseroot', type = str, default = '/dataset/Train/D08')
    parser.add_argument('--baseroot_test', type = str, default = "/dataset/Test/D08", help = 'the testing folder')
    parser.add_argument('--dataset', type = int, default = 8, help='MSD08 or LITS')


    #--------------------------------------------------------------------------------------------------------------------------------------------    
    
    
    opt = parser.parse_args()
    print(opt)
    
    # Enter main function
    import segment_tester
    import segment_utils
    import time

    generator_path = '/models/generator.pth'
    segmentor_path = '/models/segmentor.pth'
    reconstructor_path = '/models/reconstructor.pth'

    threshold = 0.5
    segment_tester.seg_tester_co_ttt(opt, checkpoint_name='TTT', lr=1e-5, iters=10, generator_path=generator_path, segmentor_path=segmentor_path, reconstructor_path=reconstructor_path, threshold=threshold)
