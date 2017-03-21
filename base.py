import os, argparse, cPickle
from shutil import copyfile
import numpy as np
import cPickle
from nibabel import load as load_nii
import nibabel as nib
from scipy import ndimage
import scipy.io as io
from data_creation import load_patches, load_only_names, load_patch_batch, get_atlas_vectors
from nolearn.lasagne import NeuralNet, BatchIterator, TrainSplit
from nolearn_utils.hooks import SaveTrainingHistory, PlotTrainingHistory, EarlyStopping
from lasagne import objectives, updates
from nolearn.lasagne.handlers import SaveWeights
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer, FeaturePoolLayer, LocalResponseNormalization2DLayer, BatchNormLayer, prelu, ConcatLayer, ElemwiseSumLayer
from lasagne.layers.dnn import Conv3DDNNLayer, MaxPool3DDNNLayer, Conv2DDNNLayer, MaxPool2DDNNLayer, Pool3DDNNLayer, batch_norm_dnn
from lasagne.nonlinearities import softmax, rectify
nib.Nifti1Header.quaternion_threshold = -np.finfo(np.float32).eps * 10


def load_data(options):
    """
    Extact data from all images.  For all database image, patches from each image view (axial, coronal and saggital) are computed.
    This function is used to reduce the loading time in leave-one-out. So, data is only loaded one time and then training feature vectors
    for the classification of each image (leave-one-out or others) are computed. 

    Input: 
    - options:
        - training folder (folder)
        - T1 name 
        - label name (mask)
        - patch size [p1, p2]

    - output:
      - x_axial: a list of X data (axial slice) indexed for each of the database subjects. Each element contains array of [features, p1, p2]
      - y_axial: a list of labels (axial slice) indexed for each of the database subjects. Each element contains array of [features, p1, p2]
      - ...
      - y_saggital: a list of labels (saggital slice) indexed for each of the database subjects. Each element contains array of [features, p1, p2]
      - centers: a list of voxel coordinates for each of the extracted patches for each of the database subjects. 
      - image names 
    """
    
    (x_axial, y_axial, x_cor, y_cor, x_sag, y_sag, centers, names) = load_patches(
        dir_name=options['folder'],
        t1_name=options['t1'],
        mask_name=options['mask'],
        size=tuple(options['patch_size']))
    return x_axial, y_axial, x_cor, y_cor, x_sag, y_sag, centers, names


def load_names(options):
    """
    Load image names. Extract names from folders and return a list. 
    """
    names = load_only_names(
        dir_name=options['folder'],
        use_t1=options['use_t1'],
        t1_name=options['t1'],
        mask_name=options['mask'],
        size=tuple(options['patch_size'])
    )
    return names



def k_fold_cross_validation_training(x_axial, y_axial, x_cor, y_cor, x_sag, y_sag, all_centers, subject_names, options):
    '''
    Perform k-fold-cross classification of the entire database of images. This function is called once and 
    we iterate through the images. For each image, a training feature vector X and Y is computed removing the current
    image. So far, a quick cascade implementation:

    [train data] --> [train_cnn1] --> [test on training data] --> [extract FP as new training data] --> [CNN2]

    Once the CNN1 is trained, all images used for training have to be recalled again and tested, as the model does not train using the whole
    training set, but all positive (sub-cortical classes) and the same number of negatives (background)

    Inputs: 
      - x_axial: a list of X data (axial slice) indexed for each of the database subjects. Each element contains array of [features, p1, p2]
      - y_axial: a list of labels (axial slice) indexed for each of the database subjects. Each element contains array of [features, p1, p2]
      - ...
      - y_saggital: a list of labels (saggital slice) indexed for each of the database subjects. Each element contains array of [features, p1, p2]
      - centers: a list of voxel coordinates for each of the extracted patches for each of the database subjects. 
      - image names 
      - options for different options in leave-one-out training. Now incorporating also the options['levels'], which controls how many cascade
        levels are performed
        options['k-fold'] controls the k-fold validation ratio.
    '''
    
    k = options['k-fold']
    for i in range(0,len(subject_names),k):

        if i == 0:
            continue
        
        # organize experiments
        current_scan = os.path.split(os.path.split(subject_names[i])[0])[-1]
        print "\n--------------------------------------------------"
        print "training on subject :", current_scan
        print "--------------------------------------------------\n"
        experiment = options['experiment']
        if options['organize_experiments']:
            exp_folder = os.path.join(options['folder'], current_scan, experiment)
            if not os.path.exists(exp_folder):
                os.mkdir(exp_folder)
                os.mkdir(os.path.join(exp_folder,'nets'))
                os.mkdir(os.path.join(exp_folder,'.train'))
        else:
            exp_folder = os.path.join(options['folder'], current_scan)
            if not os.path.exists(os.path.join(exp_folder,'nets')):
                os.mkdir(os.path.join(exp_folder,'nets'))
            if not os.path.exists(os.path.join(exp_folder,'.train')):
                os.mkdir(os.path.join(exp_folder,'train'))

        # iterate through levels
        # at the end of each iterations, reload data with the worst examples

        x_axial_ = x_axial
        x_cor_ = x_cor
        x_sag_ = x_sag
        y_axial_ = y_axial
        y_cor_ = y_cor
        y_sag_ = y_sag
                        
        centers = all_centers
        for level in range(options['levels']):

            
            print '--------------------------------------------------'
            print 'LEVEL ', level +1
            print '--------------------------------------------------\n'
            

            # Build the network, if selected, load previous weights
            # If resampling positive classes, resample each "epochs_by_sample"
            
            if options['re-sampling']:
                options['max_epochs'] = options['epochs_by_sample']
                
            print current_scan + ' Build the model'
            net = build_model(os.path.join(options['folder'], current_scan), options, level = level)

            if options['load_weights'] == True:
                try:
                    net_weights = os.path.join(exp_folder, 'nets', options['weights_name'][level])
                    net.load_params_from(net_weights)
                except:
                    print  current_scan, 'No network weights available. Training from scratch.'

            if options['re-sampling']:
                iterations = options['max_epochs'] / options['epochs_by_sample']
                for it in range(iterations):

                    # load training data for the current scan
                    x_train_axial, x_train_cor, x_train_sag, train_atlas, y_train = generate_training_set(options['folder'], current_scan, i, x_axial_, x_cor_, x_sag_, y_axial_, centers = centers, k_fold = k)

                    print "\n--------------------------------------------------"
                    print current_scan, '---- iteration: ', it, '----'
                    print current_scan +  ': X axial: Training data = (' + ','.join([str(length) for length in x_train_axial.shape]) + ')'
                    print current_scan +  ': X cor  : Training data = (' + ','.join([str(length) for length in x_train_cor.shape]) + ')'
                    print current_scan +  ': X sag  : Training data = (' + ','.join([str(length) for length in x_train_sag.shape]) + ')'
                    print current_scan +  ': X atlas: Training data = (' + ','.join([str(length) for length in train_atlas.shape]) + ')' 
                    print current_scan +  ': Y: Training labels = (' + ','.join([str(length) for length in y_train.shape]) + ')' 
                    print "--------------------------------------------------\n"
            
                    # fit the classifier. save weights when finished
                    net.fit({'in1': x_train_axial, 'in2': x_train_cor, 'in3': x_train_sag, 'in4': train_atlas}, y_train)
            else:
                print "\n--------------------------------------------------"
                print current_scan, '---- iteration: ', it, '----'
                print current_scan +  ': X axial: Training data = (' + ','.join([str(length) for length in x_train_axial.shape]) + ')'
                print current_scan +  ': X cor  : Training data = (' + ','.join([str(length) for length in x_train_cor.shape]) + ')'
                print current_scan +  ': X sag  : Training data = (' + ','.join([str(length) for length in x_train_sag.shape]) + ')'
                print current_scan +  ': X atlas: Training data = (' + ','.join([str(length) for length in train_atlas.shape]) + ')' 
                print current_scan +  ': Y: Training labels = (' + ','.join([str(length) for length in y_train.shape]) + ')' 
                print "--------------------------------------------------\n"
                net.fit({'in1': x_train_axial, 'in2': x_train_cor, 'in3': x_train_sag, 'in4': train_atlas}, y_train)

            net_weights = os.path.join(exp_folder, 'nets', options['weights_name'][level])
            net.load_params_from(net_weights)

            # test the scan not used for training
            image_nii = load_nii(subject_names[i])
            image = np.zeros_like(image_nii.get_data())
            
            print current_scan, ': testing on --> ', test_scan
            
            for batch_axial, batch_cor, batch_sag, atlas, centers in load_patch_batch(subject_names[j],
                                                                                      options['test_batch_size'],
                                                                                      tuple(options['patch_size']),
                                                                                      dir_name = options['folder'],
                                                                                      current_scan = current_scan):

                y_pred = net.predict({'in1': batch_axial, 'in2': batch_cor, 'in3': batch_sag, 'in4': atlas})
                [x, y, z] = np.stack(centers, axis=1)
                image[x, y, z] = y_pred
            
            # save segmentation masks for debugging in '.train'.
            # for the current scan, save booth the labels and the probabilities in the "EXP_FOLDER"
            image_nii.get_data()[:] = image
            image_nii.to_filename(os.path.join(exp_folder, test_scan + '_level_' + str(level) + '.nii.gz'))

            # filter-out fp by taking only the higher area.
            # iterate for each of the classes
            filtered_mask = np.zeros_like(image)

            # load mni binary mask to guide the segmentation 
            atlas = load_nii(os.path.join(options['folder'], test_scan, 'mni_atlas', 'MNI_mask_subcortical.nii.gz')).get_data()
            for l in range(1,15):
                print "     processing label ", l
                th_label = image == l
                labels, num_labels = ndimage.label(th_label)
                label_list = np.unique(labels)
                # filter candidates by size. Only storing the biggest one

                num_elements_by_lesion = ndimage.labeled_comprehension(np.logical_and(th_label, atlas), labels, label_list, np.sum, float, 0)
                argmax = np.argmax(num_elements_by_lesion)
                
                # assign voxels to output
                current_voxels = np.stack(np.where(labels == argmax), axis =1)
                filtered_mask[current_voxels[:,0], current_voxels[:,1], current_voxels[:,2]] = l

            image_nii.get_data()[:] = filtered_mask
            image_nii.to_filename(os.path.join(exp_folder, test_scan + '_filt_level_' + str(level) + '.nii.gz'))


            
            '''
            # if selected, test the network. Running in batch to reduce the amount of RAM.
            print current_scan , 'Testing subjects ----------------------------'
            out_masks = []

            
            for j in range(len(subject_names)):

                test_scan = os.path.split(os.path.split(subject_names[j])[0])[-1]

                # dont process the last iteration if its not necessary 
                if (i is not j) and (level == options['levels']-1):
                    continue
                
                # avoid re-test if exists the mask used as seed for the next iteration
                if os.path.exists(os.path.join(exp_folder, '.train', test_scan + '_filt_level_' + str(level) + '.nii.gz')):
                    filtered_mask = load_nii(os.path.join(exp_folder, '.train', test_scan + '_filt_level_' + str(level) + '.nii.gz')).get_data()
                else:
                    image_nii = load_nii(subject_names[j])
                    image = np.zeros_like(image_nii.get_data())
                    image_proba = np.zeros([image_nii.get_data().shape[0], image_nii.get_data().shape[1], image_nii.get_data().shape[2], 15])
                    print current_scan, ': testing on --> ', test_scan

                    for batch_axial, batch_cor, batch_sag, atlas, centers in load_patch_batch(subject_names[j], options['test_batch_size'], tuple(options['patch_size']),
                                                                                              dir_name = options['folder'],
                                                                                              current_scan = current_scan):

                        y_pred = net.predict({'in1': batch_axial, 'in2': batch_cor, 'in3': batch_sag, 'in4': atlas})
                        #y_pred = net.predict({'in1': batch_axial, 'in2': batch_cor, 'in3': batch_sag})
                        [x, y, z] = np.stack(centers, axis=1)
                        image[x, y, z] = y_pred

                        # if the current scan is tested, also compute probabilities
                        if i==j:
                            y_pred_proba = net.predict_proba({'in1': batch_axial, 'in2': batch_cor, 'in3': batch_sag, 'in4': atlas})
                            #y_pred_proba = net.predict_proba({'in1': batch_axial, 'in2': batch_cor, 'in3': batch_sag})
                            for c in range(15):
                                image_proba[x, y, z, c] = y_pred_proba[:,c]


                    # save segmentation masks for debugging in '.train'.
                    # for the current scan, save booth the labels and the probabilities in the "EXP_FOLDER"
                    image_nii.get_data()[:] = image
                    image_nii.to_filename(os.path.join(exp_folder, '.train', test_scan + '_level_' + str(level) + '.nii.gz'))

                    if i==j:
                        image_nii.get_data()[:] = image
                        image_nii.to_filename(os.path.join(exp_folder, current_scan + '_level_' + str(level) + '.nii.gz'))
                        image_out = nib.Nifti1Image(image_proba, np.eye(4))
                        image_out.to_filename(os.path.join(exp_folder, current_scan + '_level_' + str(level) + '_proba.nii.gz'))

                    # filter-out fp by taking only the higher area.
                    # iterate for each of the classes
                    filtered_mask = np.zeros_like(image)

                    # load mni binary mask to guide the segmentation 
                    atlas = load_nii(os.path.join(options['folder'], test_scan, 'mni_atlas', 'MNI_mask_subcortical.nii.gz')).get_data()
                    for l in range(1,15):
                        print "     processing label ", l
                        th_label = image == l
                        labels, num_labels = ndimage.label(th_label)
                        label_list = np.unique(labels)
                        # filter candidates by size. Only storing the biggest one

                        num_elements_by_lesion = ndimage.labeled_comprehension(np.logical_and(th_label, atlas), labels, label_list, np.sum, float, 0)
                        argmax = np.argmax(num_elements_by_lesion)

                        # assign voxels to output
                        current_voxels = np.stack(np.where(labels == argmax), axis =1)
                        filtered_mask[current_voxels[:,0], current_voxels[:,1], current_voxels[:,2]] = l

                    image_nii.get_data()[:] = filtered_mask
                    image_nii.to_filename(os.path.join(exp_folder, '.train', test_scan + '_filt_level_' + str(level) + '.nii.gz'))

                # apped a binary mask of the segmentation ouput to used as seed 
                out_masks.append(filtered_mask > 0)
                
            # reload the training data using the worst examples as a seeds
            x_axial_, y_axial_, x_cor_, y_cor_, x_sag_, y_sag_, centers, names = load_patches(dir_name=options['folder'],
                                                                                        t1_name=options['t1'],
                                                                                        mask_name=options['mask'],
                                                                                        size=tuple(options['patch_size']),
                                                                                        seeds = out_masks)

            '''
            
def test_all_scans(subject_names, options):
    """
    Perform testing on all the scans of the database. It assumes that for each subject, a trained 
    network already exists. Usng a cascade approach. 

    Input: 
    - subject_names: a list containing the names of each of the subjects of the database.
    - options file for testing 
    """
    for i in range(len(subject_names[0])):

        # organize experiments

        current_scan = os.path.split(os.path.split(subject_names[i])[0])[-1]
        print "--- testing on subject :", current_scan, '---------'
        
        experiment = options['experiment']
        if options['organize_experiments']:
            exp_folder = os.path.join(options['folder'], current_scan, experiment)
            if not os.path.exists(exp_folder):
                os.mkdir(exp_folder)
                os.mkdir(os.path.join(exp_folder,'nets'))
                os.mkdir(os.path.join(exp_folder,'.train'))
        else:
            exp_folder = os.path.join(options['folder'], current_scan)
            if not os.path.exists(os.path.join(exp_folder,'nets')):
                os.mkdir(os.path.join(exp_folder,'nets'))
            if not os.path.exists(os.path.join(exp_folder,'.train')):
                os.mkdir(os.path.join(exp_folder,'train'))

        # itereate between iterations

        positive_samples = None
        test_scan = os.path.split(os.path.split(subject_names[i])[0])[-1]
        image_nii = load_nii(subject_names[i])
        
        for level in range(options['levels']):
            # build the network model for the particular image 
            # load previous weights for testing

            net = build_model(os.path.join(options['folder'], current_scan), options, level = level)
            net_weights = os.path.join(exp_folder, 'nets', options['weights_name'][level])
            net.load_params_from(net_weights)

            image = np.zeros_like(image_nii.get_data())
            image_proba = np.zeros([image_nii.get_data().shape[0], image_nii.get_data().shape[1], image_nii.get_data().shape[2], 15])
            print "debug image proba: ", image_proba.shape
            print current_scan, ': testing on --> ', test_scan, ' (level ', level , ')'

            for batch_axial, batch_cor, batch_sag, atlas, centers in load_patch_batch(subject_names[i], options['test_batch_size'], tuple(options['patch_size']),
                                                                                              dir_name = options['folder'],
                                                                                              current_scan = current_scan):

                print current_scan, batch_axial.shape
                # predict classes
                y_pred = net.predict({'in1': batch_axial, 'in2': batch_cor, 'in3': batch_sag, 'in4': atlas})
                #y_pred = net.predict({'in1': batch_axial, 'in2': batch_cor, 'in3': batch_sag})
                [x, y, z] = np.stack(centers, axis=1)
                image[x, y, z] = y_pred

                # predict probabilities 
                y_pred_proba = net.predict_proba({'in1': batch_axial, 'in2': batch_cor, 'in3': batch_sag, 'in4': atlas})
                #y_pred_proba = net.predict_proba({'in1': batch_axial, 'in2': batch_cor, 'in3': batch_sag})
                for c in range(15):
                    image_proba[x, y, z, c] = y_pred_proba[:,c]

            # save segmentations 
            image_nii.get_data()[:] = image
            image_nii.to_filename(os.path.join(exp_folder,  test_scan + '_level_' + str(level) + '.nii.gz'))
            image_out = nib.Nifti1Image(image_proba, np.eye(4))
            image_out.to_filename(os.path.join(exp_folder,  test_scan + '_level_' + str(level) + '_proba.nii.gz'))

            # filter-out fp by taking only the higher area.
            # iterate for each of the classes
            filtered_mask = np.zeros_like(image)            
            atlas = load_nii(os.path.join(options['folder'], test_scan, 'mni_atlas', 'MNI_mask_subcortical.nii.gz')).get_data()
            for l in range(1,15):
                print "     processing label ", l
                th_label = image == l
                labels, num_labels = ndimage.label(th_label)
                label_list = np.unique(labels)
                # filter candidates by size. Only storing the biggest one

                num_elements_by_lesion = ndimage.labeled_comprehension(np.logical_and(th_label, atlas), labels, label_list, np.sum, float, 0)
                argmax = np.argmax(num_elements_by_lesion)

                # assign voxels to output
                current_voxels = np.stack(np.where(labels == argmax), axis =1)
                filtered_mask[current_voxels[:,0], current_voxels[:,1], current_voxels[:,2]] = l
    
            image_nii.get_data()[:] = filtered_mask
            image_nii.to_filename(os.path.join(exp_folder, test_scan + '_filt_level_' + str(level) + '.nii.gz'))

            # take positive samples from the segmentation and used as seed for the next iteration 
            #positive_samples = filtered_mask > 0

                
def generate_training_set(training_folder, current_scan, index, x_axial, x_coronal, x_saggital, y, centers = None, randomize = True, k_fold = 1):
    """
    Generate training features X an Y for each image modality. Remove the current scan "i" and build the training 
    vector. 

    input: 
    - training folder: used to load the atlas
    - current_scan: used to load the atlas
    - i: index of the current scan (to remove from training)
    - x_axial: a list of X data (axial slice) indexed for each of the database subjects. Each element contains array of [features, p1, p2]
    - y_axial: a list of labels (axial slice) indexed for each of the database subjects. Each element contains array of [features, p1, p2]
    - ...
    - y_saggital: a list of labels (saggital slice) indexed for each of the database subjects. Each element contains array of [features, p1, p2]
    - centers: voxel positions for each patch (same size than x_axial et al.)
    - randomize: randomize training vectors
    - k_fold: integer controlling the size "k" of the k-fold cross validation

    output:
    - x_train_axial: axial training X   [num_samples, num_channels, p1, p2] 
    - x_train_cor: coronal training X   [num_samples, num_channels, p1, p2] 
    - x_train_sag: saggital training X  [num_samples, num_channels, p1, p2] 
    - y_train: labels. 
           * If voxelwise classification: [num_samples]
           * If fully-convolutional: [num_samples, 1, p1, p2] 
    - atlas: atlas vector containing the probability for each of the samples
    """

    # control the number of samples used for training 

    # generate a training set by leaving out the scan used for training 
    x_train_axial = x_axial[:index] + x_axial[index+k_fold:]
    x_train_cor = x_coronal[:index] + x_coronal[index+k_fold:]
    x_train_sag = x_saggital[:index] + x_saggital[index+k_fold:]
    
    # using voxelwise segmentation. so only one segmentation is needed.
    y_train = y[:index] + y[index+k_fold:]

    atlas = get_atlas_vectors(training_folder, current_scan, centers)
    train_atlas = atlas[:index] + atlas[index+k_fold:]

    # randomize the training set
    if randomize:
        seed = np.random.randint(np.iinfo(np.int32).max)
        np.random.seed(seed)
        x_train_axial = np.random.permutation(np.concatenate(x_train_axial).astype(dtype=np.float32))
        np.random.seed(seed)
        x_train_cor = np.random.permutation(np.concatenate(x_train_cor).astype(dtype=np.float32))
        np.random.seed(seed)
        x_train_sag = np.random.permutation(np.concatenate(x_train_sag).astype(dtype=np.float32))
        np.random.seed(seed)
        y_train = np.random.permutation(np.concatenate(y_train).astype(dtype=np.uint8))
        np.random.seed(seed)
        train_atlas = np.random.permutation(np.concatenate(train_atlas).astype(dtype=np.uint8))

    # voxelwise:  convert labels to one-dimensional vector
    y_train = y_train[:, y_train.shape[1] / 2, y_train.shape[2] / 2]
    y_train = np.squeeze(y_train)

    y_train[y_train==15] = 0

    if options['re-sampling']:

        # balance positive classes
        # compute the frequency of each of the classes to obtain the class with less samples
        h, f = np.histogram(y_train, bins = 15)
        min_class = np.argmin(h)
        min_element = np.min(h)

        # new balanced indices with the same positves samples than negatives
        # permute indices before selecting patches
        y_indices_neg = np.where(y_train == 0)[0][:min_element*14]
        y_indices_pos = [np.where(y_train == i)[0][:min_element] for i in range(1,15)]
        y_indices_pos = np.concatenate(y_indices_pos, axis = 0)
        y_indices = np.random.permutation(np.concatenate([y_indices_neg, y_indices_pos], axis = 0))

        # assing new patch samples for training 
        y_train = y_train[y_indices]
        x_train_axial = x_train_axial[y_indices]
        x_train_cor = x_train_cor[y_indices]
        x_train_sag = x_train_sag[y_indices]
        train_atlas = train_atlas[y_indices]
        
    # The net expects training data with shape [samples, channels, p1, p2]
    # reshape arrays for single channel
    x_train_axial = np.expand_dims(x_train_axial, axis = 1)
    x_train_cor = np.expand_dims(x_train_cor, axis = 1)
    x_train_sag = np.expand_dims(x_train_sag, axis = 1)

    print "X_TRAIN: ", x_train_axial.shape[0]
    print "Y_TRAIN POS: ", y_train[y_train > 0].shape[0]
    print "Y_TRAIN NEG: ", y_train[y_train == 0].shape[0]
    
    return x_train_axial, x_train_cor, x_train_sag, train_atlas, y_train


def build_model(subject_path, options, level = 0):
    """
    Build the CNN model. Create the Neural Net object and return it back. 
    Inputs: 
    - subject name: used to save the net weights accordingly.
    - options: several hyper-parameters used to configure the net.
    - level: cascade level 
    
    Output:
    - net: a NeuralNet object 
    """
    # define paths to save weights and nets
    current_folder = subject_path
    net_model_name = options['weights_name'][level]
    
    # organize_experiments
    if options['organize_experiments']:
        net_weights = os.path.join(subject_path, options['experiment'], 'nets',  net_model_name)
        net_history  = os.path.join(subject_path, options['experiment'], 'nets', net_model_name+'_history.pkl')
    else:
        net_weights = os.path.join(subject_path, 'nets',  net_model_name)
        net_history  = os.path.join(subject_path, 'nets', net_model_name+'_history.pkl')

    # select hyper-parameters
    t_verbose = options['verbose']  
    train_split_perc = options['train_split']
    num_epochs = options['max_epochs']
    max_epochs_patience = options['patience']
    early_stopping = EarlyStopping(patience=max_epochs_patience)
    save_weights = SaveWeights(net_weights, only_best=True, pickle=False)
    save_training_history = SaveTrainingHistory(net_history)

    # build the architecture 

    ps = 32
    p_drop = 0.4
    num_filt = 40
    num_fc = 80
    num_channels = 1

    # --------------------------------------------------
    # channel_1: axial
    # --------------------------------------------------
    # input: 32
    axial_ch = InputLayer(name='in1', shape=(None, num_channels, ps, ps))
    axial_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(axial_ch, name='axial_ch_conv1', num_filters=num_filt, filter_size=3)),  name = 'axial_ch_prelu1')
    axial_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(axial_ch, name='axial_ch_conv2', num_filters=num_filt, filter_size=3)),  name = 'axial_ch_prelu2')
    axial_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(axial_ch, name='axial_ch_conv3', num_filters=num_filt, filter_size=3)),  name = 'axial_ch_prelu3')
    axial_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(axial_ch, name='axial_ch_conv4', num_filters=num_filt, filter_size=3)),  name = 'axial_ch_prelu4')
    axial_ch = MaxPool2DDNNLayer(axial_ch, name='axial_max_pool_1', pool_size=2)
    axial_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(axial_ch, name='axial_ch_conv5', num_filters=num_filt*2, filter_size=3)),  name = 'axial_ch_prelu5')
    axial_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(axial_ch, name='axial_ch_conv6', num_filters=num_filt*2, filter_size=3)),  name = 'axial_ch_prelu6')
    axial_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(axial_ch, name='axial_ch_conv7', num_filters=num_filt*2, filter_size=3)),  name = 'axial_ch_prelu7')
    axial_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(axial_ch, name='axial_ch_conv8', num_filters=num_filt*2, filter_size=3)),  name = 'axial_ch_prelu8')
    axial_ch = DenseLayer(axial_ch, name='axial_d1', num_units = num_fc)
    axial_ch = prelu(axial_ch, name = 'axial_prelu_d1')
    axial_ch = DropoutLayer(axial_ch, name = 'axial_l1drop', p=p_drop)

    # --------------------------------------------------
    # channel_1: coronal
    # --------------------------------------------------
    # input: 32
    coronal_ch = InputLayer(name='in2', shape=(None, num_channels, ps, ps))
    coronal_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(coronal_ch, name='coronal_ch_conv1', num_filters=num_filt, filter_size=3)),  name = 'coronal_ch_prelu1')
    coronal_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(coronal_ch, name='coronal_ch_conv2', num_filters=num_filt, filter_size=3)),  name = 'coronal_ch_prelu2')
    coronal_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(coronal_ch, name='coronal_ch_conv3', num_filters=num_filt, filter_size=3)),  name = 'coronal_ch_prelu3')
    coronal_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(coronal_ch, name='coronal_ch_conv4', num_filters=num_filt, filter_size=3)),  name = 'coronal_ch_prelu4')
    coronal_ch = MaxPool2DDNNLayer(coronal_ch, name='coronal_max_pool_1', pool_size=2)
    coronal_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(coronal_ch, name='coronal_ch_conv5', num_filters=num_filt*2, filter_size=3)),  name = 'coronal_ch_prelu5')
    coronal_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(coronal_ch, name='coronal_ch_conv6', num_filters=num_filt*2, filter_size=3)),  name = 'coronal_ch_prelu6')
    coronal_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(coronal_ch, name='coronal_ch_conv7', num_filters=num_filt*2, filter_size=3)),  name = 'coronal_ch_prelu7')
    coronal_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(coronal_ch, name='coronal_ch_conv8', num_filters=num_filt*2, filter_size=3)),  name = 'coronal_ch_prelu8')
    coronal_ch = DenseLayer(coronal_ch, name='coronal_d1', num_units = num_fc)
    coronal_ch = prelu(coronal_ch, name = 'coronal_prelu_d1')
    coronal_ch = DropoutLayer(coronal_ch, name = 'coronal_l1drop', p=p_drop)

    # --------------------------------------------------
    # channel_1: saggital
    # --------------------------------------------------
    # input: 32
    saggital_ch = InputLayer(name='in3', shape=(None, num_channels, ps, ps))
    saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv1', num_filters=num_filt, filter_size=3)),  name = 'saggital_ch_prelu1')
    saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv2', num_filters=num_filt, filter_size=3)),  name = 'saggital_ch_prelu2')
    saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv3', num_filters=num_filt, filter_size=3)),  name = 'saggital_ch_prelu3')
    saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv4', num_filters=num_filt, filter_size=3)),  name = 'saggital_ch_prelu4')
    saggital_ch = MaxPool2DDNNLayer(saggital_ch, name='saggital_max_pool_1', pool_size=2)
    saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv5', num_filters=num_filt*2, filter_size=3)),  name = 'saggital_ch_prelu5')
    saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv6', num_filters=num_filt*2, filter_size=3)),  name = 'saggital_ch_prelu6')
    saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv7', num_filters=num_filt*2, filter_size=3)),  name = 'saggital_ch_prelu7')
    saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv8', num_filters=num_filt*2, filter_size=3)),  name = 'saggital_ch_prelu8')
    saggital_ch = DenseLayer(saggital_ch, name='saggital_d1', num_units = num_fc)
    saggital_ch = prelu(saggital_ch, name = 'saggital_prelu_d1')
    saggital_ch = DropoutLayer(saggital_ch, name = 'saggital_l1drop', p=p_drop)
    
    # concatenate channels
    atlas_layer = InputLayer(name='in4', shape=(None, 15))
    layer = ConcatLayer(name = 'elem_channels', incomings = [axial_ch, coronal_ch, saggital_ch, atlas_layer])
    
    # fully connected layer
    fc_units = num_fc * 3 + 15
    layer = DenseLayer(layer, name='fc_2', num_units = fc_units)
    layer = prelu(layer, name = 'prelu_f2')
    layer = DropoutLayer(layer, name = 'f2_drop', p=p_drop)

    # fully connected layer
    layer = DenseLayer(layer, name='fc_3', num_units = fc_units)
    layer = prelu(layer, name = 'prelu_f3')
    layer = DropoutLayer(layer, name = 'f2_drop', p=p_drop)

    # softmax
    net_layer = DenseLayer(layer, name='out_layer', num_units = 15, nonlinearity=softmax)

    net =  NeuralNet(
        layers= net_layer,
        objective_loss_function=objectives.categorical_crossentropy,
        update = updates.adadelta,
        on_epoch_finished=[
            save_weights,
            save_training_history,
            early_stopping,
        ],
        verbose= t_verbose,
        max_epochs= num_epochs,
        train_split=TrainSplit(eval_size= train_split_perc),
    )
    
    return net


def test_2d_patches(x_axial, x_coronal, x_saggital, scan, patch_num):
    """
    Silly function to test patch creation. 
    Plots pngs for testing.

    Inputs: 
    - x_axial: axial patches [num_image, samples, p1, p2]
    - x_coronal: coronal patches [num_image, samples, p1, p2]
    - x_saggital: sagital patches [num_image, samples, p1, p2]

    """
    import scipy.misc
    scipy.misc.imsave('axial.png', x_axial[scan][patch_num])
    scipy.misc.imsave('coronal.png', x_coronal[scan][patch_num])
    scipy.misc.imsave('saggital.png', x_saggital[scan][patch_num])

