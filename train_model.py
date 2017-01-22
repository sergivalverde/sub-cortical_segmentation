from base import load_data, load_names, leave_one_out_training, test_all_scans


# ------------------------------------------------------------------------------------
# select hyper-parameters
# ------------------------------------------------------------------------------------
options = {}
# - experiment name
experiment = 'ibsr18_cascade_29sum'       
options['experiment'] = experiment
# - select if save models and masks in separate folders
options['organize_experiments'] = True
# - input data folder
options['folder'] = '/mnt/DATA/w/CNN_CORT/images/IBSR18'
# - CNN weights and architecture name (for saving)
options['weights_name'] = [experiment + '_1.pkl', experiment + '_2.pkl']
# - t1 name
options['use_t1'] = True
options['t1'] = 'T1.nii.gz'
# - label name
options['mask'] = 'gt_all_subcortical.nii.gz'
# - output segmentation name 
options['out_mask'] = 'CNN_' + experiment
# - number of CNN channels in the feature vector
options['num_channels'] = 1
# - CNN patch size
options['patch_size'] = [29,29]
# - CNN training batch size
options['batch_size'] = 256
# - Number of maximum iterations permitted without reducing the validation loss
options['patience'] = 25
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
options['levels'] = 2

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
        test_all_scans(subject_names, options)


