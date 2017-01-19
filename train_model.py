from base import load_data, load_names, leave_one_out_training, test_all_scans


# ------------------------------------------------------------------------------------
# select hyper-parameters
# ------------------------------------------------------------------------------------
options = {}
# - experiment name
experiment = 'ibsr18_2D_multi'       
options['experiment'] = experiment
# - select if save models and masks in separate folders
options['organize_experiments'] = True
# - input data folder
options['folder'] = '/mnt/DATA/w/CNN_CORT/images/IBSR18'
# - CNN weights and architecture name (for saving)
options['weights_name'] = experiment + '.pkl'
# - t1 name
options['t1'] = 'T1.nii.gz'
# - label name
options['mask'] = 'gt_15_classes.nii.gz'
# - output segmentation name 
options['out_mask'] = 'CNN_' + experiment
# - number of CNN channels in the feature vector
options['num_channels'] = 1
# - CNN patch size
options['patch_size'] = [29,29]
# - CNN training batch size
options['batch_size'] = 256
# - Number of maximum iterations permitted without reducing the validation loss
options['patience'] = 10
# - verbosity of the training process  
options['verbose'] = 11
# - max epochs to train 
options['max_epochs'] = 200
# - percentage of training / validation (train_split)
options['train_split'] = 0.25
# - testing batch size
options['test_batch_size'] = 50000
# - permission to reload weights (load_weights)
options['load_weights'] = True
# - Switch between training or testing the whole dataset 
options['training'] = True
# - Test excluded image during leave one out training 
options['testing'] = True
# -------------------------------------------------------------------------------------


# main script for leave-one-out training 
if __name__ == '__main__':
    if options['training'] is True:
        # load feature data and perform leave-one-out training 
        x_axial, y_axial, x_cor, y_cor, x_sag, y_sag, centers, subject_names = load_data(options)
        leave_one_out_training(x_axial, y_axial, x_cor, y_cor, x_sag, y_sag, centers, subject_names, options)
    else:
        # if training is disabled, test all the images of the dataset using existing weights, assuming that those exits
        subject_names = load_names(options)
        leave_one_out_testing(subject_names, options)


