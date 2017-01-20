import os, argparse, cPickle
from shutil import copyfile
import numpy as np
import cPickle
from nibabel import load as load_nii
from scipy import ndimage
import scipy.io as io
from data_creation import load_patches, load_only_names, load_patch_batch
from nolearn.lasagne import NeuralNet, BatchIterator, TrainSplit
from nolearn_utils.hooks import SaveTrainingHistory, PlotTrainingHistory, EarlyStopping
from lasagne import objectives, updates
from nolearn.lasagne.handlers import SaveWeights

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

def leave_one_out_training(x_axial, y_axial, x_cor, y_cor, x_sag, y_sag, centers, subject_names, options):
    '''
    Perform leave-one-out classification of the entire database of images. This function is called once and 
    we iterate through the images. For each image, a training feature vector X and Y is computed removing the current
    image. 

    Inputs: 
      - x_axial: a list of X data (axial slice) indexed for each of the database subjects. Each element contains array of [features, p1, p2]
      - y_axial: a list of labels (axial slice) indexed for each of the database subjects. Each element contains array of [features, p1, p2]
      - ...
      - y_saggital: a list of labels (saggital slice) indexed for each of the database subjects. Each element contains array of [features, p1, p2]
      - centers: a list of voxel coordinates for each of the extracted patches for each of the database subjects. 
      - image names 
      - options for different options in leave-one-out training 
    '''
    
    for i in range(len(subject_names)):

        # organize experiments
        current_scan = os.path.split(os.path.split(subject_names[0])[0])[-1]
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

        # load training data for the current scan
        x_train_axial, x_train_cor, x_train_sag, y_train = generate_training_set(i, x_axial, x_cor, x_sag, y_axial)

        print "\n--------------------------------------------------"
        print current_scan +  ': X axial: Training data = (' + ','.join([str(length) for length in x_train_axial.shape]) + ')'
        print current_scan +  ': X cor  : Training data = (' + ','.join([str(length) for length in x_train_cor.shape]) + ')'
        print current_scan +  ': X sag  : Training data = (' + ','.join([str(length) for length in x_train_sag.shape]) + ')' 
        print current_scan +  ': Y: Training labels = (' + ','.join([str(length) for length in y_train.shape]) + ')' 
        print "--------------------------------------------------\n"

        # build the network model 
        # if selected, load previous weights
        print current_scan + ' Build the model'
        net = build_model(os.path.join(options['folder'], current_scan), options)

        if options['load_weights'] == True:
            try:
                net_weights = os.path.join(exp_folder, 'nets', options['weights_name'])
                net.load_params_from(net_weights)
            except:
                print  current_scan, 'No network weights available. Training from scratch.'
                    

        # fit the classifier. save weights when finished
        net.fit({'in1': x_train_axial, 'in2': x_train_cor, 'in3': x_train_sag}, y_train)
       
        net_weights = os.path.join(exp_folder, 'nets', options['net_model'])
        net.load_params_from(net_weights)

        # if selected, test the network. Running in batch to reduce the amount of RAM.
        if options['testing']:
            print current_scan , 'Testing subject ----------------------------'

            image_nii = load_nii(subject_names[0, i])
            image = np.zeros_like(image_nii.get_data())
            
            for batch_axial, batch_cor, batch_sag, centers, atlas in load_patch_batch(subject_names[:, i], options['test_batch_size'], tuple(options['patch_size'])):
                y_pred = net.predict({'in1': batch_axial, 'in2': batch_cor, 'in3': batch_sag})
                [x, y, z] = np.stack(centers, axis=1)
                image[x, y, z] = np.expand_dims(y_pred, axis = 1)
            image_nii.get_data()[:] = image
            image_nii.to_filename(os.path.join(exp_folder, options['out_mask'] +'_1.nii.gz'))

    
def test_all_scans(subject_names, options):
    """
    Perform testing on all the scans of the database. It assumes that for each subject, a trained 
    network already exists. 

    Input: 
    - subject_names: a list containing the names of each of the subjects of the database.
    - options file for testing 
    """
    for i in range(len(subject_names[0])):

        # organize experiments

        current_scan = os.path.split(os.path.split(subject_names[0,i])[0])[-1]
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

        
        # build the network model for the particular image 
        # load previous weights for testing
                   
        print current_scan + ' Build the model'
        net = build_model(os.path.join(options['folder'], current_scan), options)
        net_weights = os.path.join(exp_folder, 'nets', options['weights_name'])
        net.load_params_from(net_weights)

        # test 
        image_nii = load_nii(subject_names[0, i])
        image = np.zeros_like(image_nii.get_data())
            
        for batch_axial, batch_cor, batch_sag, centers in load_patch_batch(subject_names[:, i], options['test_batch_size'], tuple(options['patch_size'])):
            y_pred = net.predict({'in1': batch_axial, 'in2': batch_cor, 'in3': batch_sag})
            [x, y, z] = np.stack(centers, axis=1)
            image[x, y, z] = np.expand_dims(y_pred, axis = 1)

        image_nii.get_data()[:] = image
        image_nii.to_filename(os.path.join(exp_folder, options['out_mask'] +'_1.nii.gz'))


                
def generate_training_set(index, x_axial, x_coronal, x_saggital, y, centers = None, randomize = True):
    """
    Generate training features X an Y for each image modality. Remove the current scan "i" and build the training 
    vector. 

    input: 
    - i: index of the current scan (to remove from training)
    - x_axial: a list of X data (axial slice) indexed for each of the database subjects. Each element contains array of [features, p1, p2]
    - y_axial: a list of labels (axial slice) indexed for each of the database subjects. Each element contains array of [features, p1, p2]
    - ...
    - y_saggital: a list of labels (saggital slice) indexed for each of the database subjects. Each element contains array of [features, p1, p2]
    - centers: voxel positions for each patch (same size than x_axial et al.)
    - randomize: randomize training vectors

    output:
    - x_train_axial: axial training X   [num_samples, num_channels, p1, p2] 
    - x_train_cor: coronal training X   [num_samples, num_channels, p1, p2] 
    - x_train_sag: saggital training X  [num_samples, num_channels, p1, p2] 
    - y_train: labels. 
           * If voxelwise classification: [num_samples]
           * If fully-convolutional: [num_samples, 1, p1, p2] 
    """
    
    # generate a training set by leaving out the scan used for training 
    x_train_axial = x_axial[:index] + x_axial[index+1:]
    x_train_cor = x_coronal[:index] + x_coronal[index+1:]
    x_train_sag = x_saggital[:index] + x_saggital[index+1:]

    # using voxelwise segmentation. so only one segmentation is needed.
    y_train = y[:index] + y[index+1:]

    #atlas = get_atlas_vectors(options['folder'], current_scan, all_centers)
    #train_atlas = atlas[:i] + atlas[i+1:]

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
        #np.random.seed(seed)
        #train_atlas = np.random.permutation(np.concatenate(train_atlas).astype(dtype=np.uint8))

    # voxelwise:  convert labels to one-dimensional vector
    y_train = y_train[:, y_train.shape[1] / 2, y_train.shape[2] / 2]
    y_train = np.squeeze(y_train)

    # The net expects training data with shape [samples, channels, p1, p2]
    # reshape arrays for single channel
    x_train_axial = np.expand_dims(x_train_axial, axis = 1)
    x_train_cor = np.expand_dims(x_train_cor, axis = 1)
    x_train_sag = np.expand_dims(x_train_sag, axis = 1)

    # IBSR related: a selected background around structs is selected in label mask to separate from the rest
    # of background. background has label = 15 but should be 0.
    y_train[y_train==15] = 0
    return x_train_axial, x_train_cor, x_train_sag, y_train


def build_model(subject_path, options):
    """
    Build the CNN model. Create the Neural Net object and return it back. 
    Inputs: 
    - subject name: used to save the net weights accordingly.
    - options: several hyper-parameters used to configure the net.
    
    Output:
    - net: a NeuralNet object 
    """
    # define paths to save weights and nets
    current_folder = subject_path
    net_model_name = options['weights_name']
    
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

    # loading net_layers from config file --> net_layer
    from build_model import net_layer

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

