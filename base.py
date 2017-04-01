import os, argparse, cPickle
from shutil import copyfile
import numpy as np
import cPickle
from nibabel import load as load_nii
import nibabel as nib
from scipy import ndimage
import scipy.io as io
from data_creation import load_patches, load_only_names, load_patch_batch, get_atlas_vectors
from nets import build_model
from skimage.transform import SimilarityTransform, warp, AffineTransform, rotate 
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
    
    (x_axial, y_axial, x_cor, y_cor, x_sag, y_sag, centers, names) = load_patches(dir_name=options['folder'],
                                                                                  t1_name=options['t1'],
                                                                                  mask_name=options['mask'],
                                                                                  size=tuple(options['patch_size']),
                                                                                  balance_neg = options['balance_neg'])
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
                iterations = options['max_epochs'] / options['epochs_by_sample']
                options['max_epochs'] = options['epochs_by_sample']

                print "Re-sampling is enabled...."
                print "Num iterations:", iterations, "max epochs by iteration" , options['max_epochs']

            print current_scan + ' Build the model'
            net = build_model(os.path.join(options['folder'], current_scan), options, level = level)

            if options['load_weights'] == True:
                try:
                    net_weights = os.path.join(exp_folder, 'nets', options['weights_name'][level])
                    net.load_params_from(net_weights)
                except:
                    print  current_scan, 'No network weights available. Training from scratch'

            if options['re-sampling']:
                print iterations
                for it in range(iterations):

                    # load training data for the current scan
                    x_train_axial, x_train_cor, x_train_sag, train_atlas, y_train = generate_training_set(options['folder'], current_scan, i, x_axial_, x_cor_, x_sag_, y_axial_, options, centers = centers, k_fold = k)

                    if it == 0:                
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
                x_train_axial, x_train_cor, x_train_sag, train_atlas, y_train = generate_training_set(options['folder'], current_scan, i, x_axial_, x_cor_, x_sag_, y_axial_, options, centers = centers, k_fold = k)
            
                print "\n--------------------------------------------------"
                print current_scan +  ': X axial: Training data = (' + ','.join([str(length) for length in x_train_axial.shape]) + ')'
                print current_scan +  ': X cor  : Training data = (' + ','.join([str(length) for length in x_train_cor.shape]) + ')'
                print current_scan +  ': X sag  : Training data = (' + ','.join([str(length) for length in x_train_sag.shape]) + ')'
                print current_scan +  ': X atlas: Training data = (' + ','.join([str(length) for length in train_atlas.shape]) + ')' 
                print current_scan +  ': Y: Training labels = (' + ','.join([str(length) for length in y_train.shape]) + ')' 
                print "--------------------------------------------------\n"
                # fit the model 
                net.fit({'in1': x_train_axial, 'in2': x_train_cor, 'in3': x_train_sag, 'in4': train_atlas}, y_train)

            net_weights = os.path.join(exp_folder, 'nets', options['weights_name'][level])
            net.load_params_from(net_weights)

            if options['testing']:
                # test the scan not used for training
                image_nii = load_nii(subject_names[i])
                image = np.zeros_like(image_nii.get_data())
            
                print current_scan, ': testing on --> ', current_scan
                
                for batch_axial, batch_cor, batch_sag, atlas, centers in load_patch_batch(subject_names[i],
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
                atlas = load_nii(os.path.join(options['folder'], current_scan, 'mni_atlas', 'MNI_mask_subcortical.nii.gz')).get_data()
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
                image_nii.to_filename(os.path.join(exp_folder, current_scan + '_filt_level_' + str(level) + '.nii.gz'))

            
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

                if options['out_probabilities']:
                    # predict probabilities 
                    y_pred_proba = net.predict_proba({'in1': batch_axial, 'in2': batch_cor, 'in3': batch_sag, 'in4': atlas})
                    for c in range(15):
                        image_proba[x, y, z, c] = y_pred_proba[:,c]

            # save segmentations 
            image_nii.get_data()[:] = image
            image_nii.to_filename(os.path.join(exp_folder,  test_scan + '_level_' + str(level) + '.nii.gz'))
            if options['out_probabilities']:
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

                

def generate_training_set(training_folder, current_scan, index, x_axial, x_coronal, x_saggital, y, options, centers = None, randomize = True, k_fold = 1):
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
    x_train_axial = np.concatenate(x_axial[:index] + x_axial[index+k_fold:], axis = 0).astype('float32')
    x_train_cor =   np.concatenate(x_coronal[:index] + x_coronal[index+k_fold:], axis = 0).astype('float32')
    x_train_sag =   np.concatenate(x_saggital[:index] + x_saggital[index+k_fold:], axis = 0).astype('float32')

    
    # using voxelwise segmentation. so only one segmentation is needed.
    y_train = np.concatenate(y[:index] + y[index+k_fold:]).astype('uint8')

    # voxelwise:  convert labels to one-dimensional vector
    y_train = y_train[:, y_train.shape[1] / 2, y_train.shape[2] / 2]
    y_train = np.squeeze(y_train)

    y_train[y_train==15] = 0

    atlas = get_atlas_vectors(training_folder, current_scan, centers)
    train_atlas = np.concatenate(atlas[:index] + atlas[index+k_fold:]).astype('float32')

    # plot class frequencies
    

    if options['data_augmentation']:
        
        # balance positive classes
        # compute the frequency of each of the classes to obtain the class with less samples



        h, f = np.histogram(y_train, bins = 15)
        min_class = np.argmin(h)
        min_element = np.min(h)
        print "--------------------------------------------------"
        print "DATA DISTRIBUTION before DA"
        print "--------------------------------------------------"
        max_element = h[1]
        for c in range(15): print "CLASS: ", c, h[c], max_element / h[c]
        
        x_train_axial, x_train_cor, x_train_sag, train_atlas, y_train = data_augmentation(x_train_axial,
                                                                                          x_train_cor,
                                                                                          x_train_sag,
                                                                                          train_atlas,
                                                                                          y_train, options)
        h, f = np.histogram(y_train, bins = 15)
        min_class = np.argmin(h)
        min_element = np.min(h)
        print "--------------------------------------------------"
        print "DATA DISTRIBUTION before DA"
        print "--------------------------------------------------"        
        max_element = h[1]
        for c in range(15): print "CLASS: ", c, h[c], max_element / h[c]

    if randomize:
        seed = np.random.randint(np.iinfo(np.int32).max)
        np.random.seed(seed)
        x_train_axial = np.random.permutation(x_train_axial)
        np.random.seed(seed)
        x_train_cor = np.random.permutation(x_train_cor)
        np.random.seed(seed)
        x_train_sag = np.random.permutation(x_train_sag)
        np.random.seed(seed)
        y_train = np.random.permutation(y_train)
        np.random.seed(seed)
        train_atlas = np.random.permutation(train_atlas)
    
            
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

    #print "X_TRAIN: ", x_train_axial.shape[0]
    #print "Y_TRAIN POS: ", y_train[y_train > 0].shape[0]
    #print "Y_TRAIN NEG: ", y_train[y_train == 0].shape[0]
    
    return x_train_axial, x_train_cor, x_train_sag, train_atlas, y_train


def data_augmentation(x_axial, x_cor, x_sag, atlas, y, options):
    """
    to doc
    Simple transformations based rotation, translation and scale 

    X = [num_samples, rows, cols]
    y = [num_samples, rows, cols]

    """
    
    # classes used for data augmentation 
    classes = options['class_weights'].keys()
    class_size = options['class_weights'].values()
    
    # augment the number of elements for each of the selected classes. The number of elements for each clas
    # is controled by class size
    x_axial_ = np.concatenate([np.concatenate([x_axial[y == c] for i in range(s)], axis = 0) for c,s in zip(classes, class_size)], axis = 0)
    x_cor_ = np.concatenate([np.concatenate([x_cor[y == c] for i in range(s)], axis = 0) for c,s in zip(classes, class_size)], axis = 0)
    x_sag_ = np.concatenate([np.concatenate([x_sag[y == c] for i in range(s)], axis = 0) for c,s in zip(classes, class_size)], axis = 0)
    y_da = np.concatenate([np.concatenate([y[y == c] for i in range(s)], axis = 0) for c,s in zip(classes, class_size)], axis = 0)
    a_da = np.concatenate([np.concatenate([atlas[y == c] for i in range(s)], axis = 0) for c,s in zip(classes, class_size)], axis = 0)

    '''
    # extract patches for augmenting
    xa_ = np.concatenate([x_axial[y == c] for c in classes], axis = 0)
    xc_ = np.concatenate([x_cor[y == c] for c in classes], axis = 0)
    xs_ = np.concatenate([x_sag[y == c] for c in classes], axis = 0)
    yy_ = np.concatenate([y[y == c] for c in classes], axis = 0)
    aa_ = np.concatenate([atlas[y == c] for c in classes], axis = 0)
    da_size = options['class_size']
    # expand classes
    x_axial_ = np.concatenate([xa_ for i in range(da_size)], axis = 0)
    x_cor_ = np.concatenate([xc_ for i in range(da_size)], axis = 0)
    x_sag_ = np.concatenate([xs_ for i in range(da_size)], axis = 0)
    y_da = np.concatenate([yy_ for i in range(da_size)], axis = 0)
    a_da = np.concatenate([aa_ for i in range(da_size)], axis = 0)
    '''

    num_samples = x_axial_.shape[0]
    random_angles = np.random.randint(size = num_samples, low = -options['max_angle'], high = options['max_angle'])
    random_noise = np.random.normal(0, options['max_noise'], x_axial_.shape)


    # apply noise transformation
    x_axial_ += random_noise
    x_cor_ += random_noise
    x_sag_ += random_noise

    
    # rotate and concatenate with input data 
    x_axial_da = np.array([rotate(patch.astype('int16'), r, order = 1, mode = 'edge', preserve_range = True) for patch, r in zip(x_axial_, random_angles)])
    x_sag_da = np.array([rotate(patch.astype('int16'), r, order = 1, mode = 'edge', preserve_range = True) for patch, r in zip(x_sag_, random_angles)])
    x_cor_da = np.array([rotate(patch.astype('int16'), r, order = 1, mode = 'edge', preserve_range = True) for patch, r in zip(x_cor_, random_angles)])


    # concatenate with input data

    # flip patches if selected 
    if options['da_flip']:
        x_axial_da = np.concatenate([x_axial_da, x_axial_da[:,:,::-1]], axis = 0)
        x_cor_da = np.concatenate([x_cor_da, x_cor_da[:,:,::-1]], axis = 0)
        x_sag_da = np.concatenate([x_sag_da, x_sag_da[:,:,::-1]], axis = 0)
        y_da = np.concatenate([y_da, y_da], axis = 0)
        a_da = np.concatenate([a_da, a_da], axis = 0)

    if options['da_shuffle']:
 
        # select data augmented patches based on the amount of DA and concatenate with original data        
        perm_indices = np.random.permutation(x_axial_da.shape[0])
        x_axial_da = x_axial_da[perm_indices]
        x_cor_da = x_cor_da[perm_indices]
        x_sag_da = x_sag_da[perm_indices]
        y_da = y_da[perm_indices]
        a_da = a_da[perm_indices]

    # concatenate with original datay_da = np.concatenate([y, y_], axis = 0)
    a_da = np.concatenate([atlas, a_da], axis = 0).astype('float32')
    y_da = np.concatenate([y, y_da], axis = 0).astype('int8')
    x_axial_da = np.concatenate([x_axial, x_axial_da], axis = 0).astype('float32')
    x_cor_da = np.concatenate([x_cor, x_cor_da], axis = 0).astype('float32')
    x_sag_da = np.concatenate([x_sag, x_sag_da], axis = 0).astype('float32')


    return x_axial_da, x_cor_da, x_sag_da, a_da, y_da 


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


