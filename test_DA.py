
from base import load_data, load_names,  test_all_scans, k_fold_cross_validation_training
import sys

# ------------------------------------------------------------------------------------
# select hyper-parameters
# ------------------------------------------------------------------------------------

# select arguments 
options = {}
options['training'] = False

experiment = 'TEST_DA'
# - experiment name
options['experiment'] = experiment
# - select if save models and masks in separate folders
options['organize_experiments'] = True
options['k-fold'] = 1
# - CNN weights and architecture name (for saving)
options['weights_name'] = [experiment + '_1.pkl', experiment + '_2.pkl']
# - t1 name
options['use_t1'] = True
options['t1'] = 'T1.nii.gz'
# - label name
#options['mask'] = 'gt_all_subcortical.nii.gz'
#options['mask'] = 'gt_15_classes.nii.gz'
options['mask'] = 'gt_15_classes_border2.nii.gz'
# - output segmentation name 
options['out_mask'] = 'CNN_' + experiment
# - number of CNN channels in the feature vector
options['num_channels'] = 1
options['patch_size'] = [32,32]

# - CNN training batch size
options['batch_size'] = 256
# - Number of maximum iterations permitted without reducing the validation loss
options['patience'] = 200
# - verbosity of the training process  
options['verbose'] = 1

# - max epochs to train 
options['max_epochs'] = 200
#resampling options

options['re-sampling'] = False
options['epochs_by_sample'] = 1 

# - percentage of training / validation (train_split)
options['train_split'] = 0.25
# - testing batch size
options['test_batch_size'] = 50000
# - permission to reload weights (load_weights)o
options['load_weights'] = True
# - Test excluded image during leave one out training 
options['testing'] = True
options['levels'] = 1

options['data_augmentation'] = True 
options['classes'] = [11,12,13,14]
options['da_size'] = 6
options['max_angle'] = 6
options['max_noise'] = 0.20
options['da_shuffle'] = False
options['da_flip'] = False
#-------------------------------------------------------------------------------------

# main script for leave-one-out training 
if __name__ == '__main__':
    if options['training'] is True:
        # load feature data and perform leave-one-out training
        #options['folder'] = '/mnt/DATA/w/CNN_CORT/images/MICCAI2012/train_tests'
        options['folder'] = '/mnt/DATA/w/CNN_CORT/images/MICCAI2012/training_set'
        x_axial, y_axial, x_cor, y_cor, x_sag, y_sag, centers, subject_names = load_data(options)
        k_fold_cross_validation_training(x_axial, y_axial, x_cor, y_cor, x_sag, y_sag, centers, subject_names, options)
    else:
        # if training is disabled, test all the images of the dataset using existing weights, assuming that those exits
        options['folder'] = '/mnt/DATA/w/CNN_CORT/images/MICCAI2012/test_set'
        subject_names = load_names(options)
        test_all_scans(subject_names, options)


