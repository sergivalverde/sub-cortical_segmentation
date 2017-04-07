import os
import argparse

# parse input options 
parser = argparse.ArgumentParser(description='Subcortical segmentation CNN')
parser.add_argument('--train', action='store_true', default=False)
parser.add_argument('--experiment', type=str)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--da', action='store_true', default=False)
parser.add_argument('--re', action='store_true', default=False)
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

options['experiment'] = experiment
options['organize_experiments'] = True
options['k-fold'] = 1

# dataset 
options['use_t1'] = True
options['t1'] = 'T1.nii.gz'
#options['mask'] = 'gt_15_classes_border2.nii.gz'
options['mask'] = 'gt_15_classes.nii.gz'
options['out_mask'] = 'CNN_' + experiment
options['out_probabilities'] = True


# - CNN training batch size
#options['patch_size'] = [25,25]
options['patch_size'] = [32,32]
options['num_channels'] = 1
options['weights_name'] = [experiment + '_1.pkl', experiment + '_2.pkl']
options['batch_size'] = 256
options['patience'] = 200
options['verbose'] = 1
options['max_epochs'] = 200
options['balance_neg'] = True 
options['re-sampling'] = args.re 
options['epochs_by_sample'] = 10 
options['train_split'] = 0.25
options['test_batch_size'] = 50000
options['load_weights'] = True
options['testing'] = False
options['levels'] = 1

# DA options (experimental)
options['data_augmentation'] = args.da
options['class_weights'] = {3:1, 4:1, 7:4, 8:4, 9:1, 10:1, 11:7, 12:7, 13:12, 14:12}
options['max_angle'] = 6
options['max_noise'] = 0.20
options['da_shuffle'] = False
options['da_flip'] = False


#-------------------------------------------------------------------------------------

# main script for leave-one-out training 
if __name__ == '__main__':
    if options['training'] is True:
        # load feature data and perform leave-one-out training
        options['folder'] = '/mnt/DATA/w/CNN_CORT/images/MICCAI2012/training_set'
        #options['folder'] = '/mnt/DATA/w/CNN_CORT/images/MICCAI2012/train_tests'
        x_axial, y_axial, x_cor, y_cor, x_sag, y_sag, centers, subject_names = load_data(options)
        k_fold_cross_validation_training(x_axial, y_axial, x_cor, y_cor, x_sag, y_sag, centers, subject_names, options)
    else:
        # if training is disabled, test all the images of the dataset using existing weights, assuming that those exits
        options['folder'] = '/mnt/DATA/w/CNN_CORT/images/MICCAI2012/test_set'
        subject_names = load_names(options)
        test_all_scans(subject_names, options)


