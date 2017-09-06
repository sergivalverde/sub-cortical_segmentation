import os
import argparse

# parse input options 
parser = argparse.ArgumentParser(description='Subcortical segmentation CNN')
parser.add_argument('--train', action='store_true', default=False)
parser.add_argument('--experiment', type=str)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--da', action='store_true', default=False)
parser.add_argument('--re', action='store_true', default=False)
parser.add_argument('--bf', action='store_true', default=False)
parser.add_argument('--bd', action='store_true', default=False)
parser.add_argument('--ps', type=int, default = 32)
args = parser.parse_args()


# theano environment
os.environ['KERAS_BACKEND'] = 'theano'
os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=gpu'+str(args.gpu)+',floatX=float32,optimizer=fast_compile'

from base import load_data, load_names,  test_all_scans, k_fold_cross_validation_training
import sys

# ------------------------------------------------------------------------------------
# select hyper-parameters
# ------------------------------------------------------------------------------------

# select arguments 
options = {}
options['training'] = args.train

# - experiment name
experiment = args.experiment

if args.da:
    experiment = experiment + '_DA'
    
if args.re:
    experiment = experiment + '_RE'
#if args.bd is False:
#    experiment = experiment + '_NB'

if args.bf:
    experiment = experiment + '_N4'
    
print experiment 
options['experiment'] = experiment
options['organize_experiments'] = True
options['k-fold'] = 1

# dataset 
options['use_t1'] = True
if args.bf:
    options['t1'] ='T1_n4.nii.gz'
else:
    options['t1'] = 'T1.nii.gz'

options['mask'] = 'gt_all_subcortical.nii.gz'
#options['mask'] = 'gt_15_classes.nii.gz'
options['out_mask'] = 'CNN_' + experiment
options['out_probabilities'] = False
# - CNN training batch size
options['patch_size'] = [args.ps, args.ps]
options['num_channels'] = 1
options['weights_name'] = [experiment + '_1.pkl', experiment + '_2.pkl']
options['batch_size'] = 256
options['patience'] = 15
options['verbose'] = 1
options['max_epochs'] = 100
options['balance_neg'] = True
options['re-sampling'] = args.re
options['resample_perc'] = 1
options['epochs_by_sample'] = 10 
options['train_split'] = 0.25
options['test_batch_size'] = 100000
options['load_weights'] = True
options['testing'] = False
options['levels'] = 1
options['tf'] = 0
options['filtering_out'] = False
options['crop'] = True 
options['atlas'] = False 

# DA options (experimental)
options['data_augmentation'] = args.da
options['class_weights'] = {3:1, 4:1, 7:4, 8:4, 9:1, 10:1, 11:4, 12:4, 13:4, 14:4}
options['max_angle'] = 6
options['max_noise'] = 0.20
options['da_shuffle'] = False
options['da_flip'] = False


#-------------------------------------------------------------------------------------

# main script for leave-one-out training 
if __name__ == '__main__':

    if options['training'] is True:
        # load feature data and perform leave-one-out training
        #options['folder'] = '/mnt/DATA/w/CNN_CORT/images/MICCAI2012/old/train_tests'
        options['folder'] = '/mnt/DATA/w/CNN_CORT/images/MICCAI2012/train'
        x_axial, y_axial, x_cor, y_cor, x_sag, y_sag, centers, subject_names = load_data(options)
        k_fold_cross_validation_training(x_axial, y_axial, x_cor, y_cor, x_sag, y_sag, centers, subject_names, options)
    else:
        # if training is disabled, test all the images of the dataset using existing weights, assuming that those exits
        options['folder'] = '/mnt/DATA/w/CNN_CORT/images/MICCAI2012/test'
        #options['folder'] = '/mnt/DATA/w/CNN_CORT/images/MICCAI2012/train'
        subject_names = load_names(options)
        test_all_scans(subject_names, options)

