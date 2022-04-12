import argparse
import os
import time

if __name__ == "__main__":

    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    
    # Path parameters
    parser.add_argument('--save_path', type = str, default = './models/', help = 'save models')
    parser.add_argument('--sample_path', type = str, default = './samples/', help = 'save samples')
    
    # Training parameters
    parser.add_argument('--epochs', type = int, default = 8, help = 'number of epochs of training')
    parser.add_argument('--batch_size', type = int, default = 4, help = 'size of the batches')
    parser.add_argument('--num_workers', type = int, default = 4, help = 'number of cpu threads to use during batch generation')
    parser.add_argument('--lr', type = float, default = 1e-4, help = 'Adam: learning rate')
    parser.add_argument('--loss_weight', type = int, default = 5,  help = 'weighted cross entropy loss for segmentation')
    parser.add_argument('--threshold', type = float, default = 0.5, help = 'threshold for binarization') 
  
    # Network parameters
    parser.add_argument('--init_type', type = str, default = 'kaiming', help = 'the initialization type')
    parser.add_argument('--init_gain', type = float, default = 0.2, help = 'the initialization gain')
   
    # tumor synthesis parameters
    parser.add_argument('--a1', type = int, default = 6, help = 'mu a')
    parser.add_argument('--b1', type = int, default = 4, help = 'mu b')
    parser.add_argument('--a2', type = int, default = 4, help = 'sigma a')
    parser.add_argument('--b2', type = int, default = 6, help = 'sigma b')


    # Dataset path
    parser.add_argument('--baseroot', type = str, default = '/dataset/Train/D08')
    parser.add_argument('--baseroot_test', type = str, default = "/dataset/Test/D08", help = 'the testing folder')

    #--------------------------------------------------------------------------------------------------------------------------------------------    
    opt = parser.parse_args()
    print(opt)


    # build folers
    c_time =  time.strftime('%Y%m%d-%H%M%S')
    print(c_time)
    opt.save_path += c_time
    opt.sample_path += c_time
    if os.path.isdir(opt.save_path) is False:
        os.mkdir(opt.save_path)
    if os.path.isdir(opt.sample_path) is False:
        os.mkdir(opt.sample_path)
    
    

    # Proposed method
    import segment_trainer

    segment_trainer.seg_trainer_ttt(opt)


