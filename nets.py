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
import lasagne
import theano as T
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer, FeaturePoolLayer, LocalResponseNormalization2DLayer, BatchNormLayer, prelu, ConcatLayer, ElemwiseSumLayer, ExpressionLayer, PadLayer, ScaleLayer
from lasagne.layers.dnn import Conv3DDNNLayer, MaxPool3DDNNLayer, Conv2DDNNLayer, MaxPool2DDNNLayer, Pool3DDNNLayer, batch_norm_dnn
from lasagne.nonlinearities import softmax, rectify
from skimage.transform import SimilarityTransform, warp, AffineTransform, rotate 
nib.Nifti1Header.quaternion_threshold = -np.finfo(np.float32).eps * 10
from datetime import datetime


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

    ps = options['patch_size'][0]
    num_channels = options['num_channels']

    net = None
    
    if (options['experiment'] == 'CONV_120_375_120_120') or (options['experiment'] == 'CONV_120_375_120_120_N4'):
    
        fc_conv = 120
        fc_fc = fc_conv
        dropout_conv = 0.4
        dropout_fc = 0.4
        # --------------------------------------------------
        # channel_1: axial
        # --------------------------------------------------
        # input: 32
        
        axial_ch = InputLayer(name='in1', shape=(None, num_channels, ps, ps))
        axial_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(axial_ch, name='axial_ch_conv1', num_filters=20, filter_size=3)),  name = 'axial_ch_prelu1')
        axial_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(axial_ch, name='axial_ch_conv2', num_filters=20, filter_size=3)),  name = 'axial_ch_prelu2')
        axial_ch = MaxPool2DDNNLayer(axial_ch, name='axial_max_pool_1', pool_size=2)
        axial_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(axial_ch, name='axial_ch_conv3', num_filters=40, filter_size=3)),  name = 'axial_ch_prelu3')
        axial_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(axial_ch, name='axial_ch_conv4', num_filters=40, filter_size=3)),  name = 'axial_ch_prelu4')
        axial_ch = MaxPool2DDNNLayer(axial_ch, name='axial_max_pool_2', pool_size=2)
        axial_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(axial_ch, name='axial_ch_conv5', num_filters=60, filter_size=3)),  name = 'axial_ch_prelu5')
        axial_ch = DenseLayer(axial_ch, name='axial_d1', num_units = fc_conv)
        axial_ch = prelu(axial_ch, name = 'axial_prelu_d1')
        axial_ch = DropoutLayer(axial_ch, name = 'axial_l1drop', p = dropout_conv)

        # --------------------------------------------------
        # channel_1: coronal
        # --------------------------------------------------
        # input: 32
        coronal_ch = InputLayer(name='in2', shape=(None, num_channels, ps, ps))
        coronal_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(coronal_ch, name='coronal_ch_conv1', num_filters=20, filter_size=3)),  name = 'coronal_ch_prelu1')
        coronal_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(coronal_ch, name='coronal_ch_conv2', num_filters=20, filter_size=3)),  name = 'coronal_ch_prelu2')
        coronal_ch = MaxPool2DDNNLayer(coronal_ch, name='coronal_max_pool_1', pool_size=2)
        coronal_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(coronal_ch, name='coronal_ch_conv3', num_filters=40, filter_size=3)),  name = 'coronal_ch_prelu3')
        coronal_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(coronal_ch, name='coronal_ch_conv4', num_filters=40, filter_size=3)),  name = 'coronal_ch_prelu4')
        coronal_ch = MaxPool2DDNNLayer(coronal_ch, name='coronal_max_pool_2', pool_size=2)
        coronal_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(coronal_ch, name='coronal_ch_conv5', num_filters=60, filter_size=3)),  name = 'coronal_ch_prelu5')
        coronal_ch = DenseLayer(coronal_ch, name='coronal_d1', num_units = fc_conv)
        coronal_ch = prelu(coronal_ch, name = 'coronal_prelu_d1')
        coronal_ch = DropoutLayer(coronal_ch, name = 'coronal_l1drop', p = dropout_conv)

        # --------------------------------------------------
        # channel_1: saggital
        # --------------------------------------------------
        #input: 32
        saggital_ch = InputLayer(name='in3', shape=(None, num_channels, ps, ps))
        saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv1', num_filters=20, filter_size=3)),  name = 'saggital_ch_prelu1')
        saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv2', num_filters=20, filter_size=3)),  name = 'saggital_ch_prelu2')
        saggital_ch = MaxPool2DDNNLayer(saggital_ch, name='saggital_max_pool_1', pool_size=2)
        saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv3', num_filters=40, filter_size=3)),  name = 'saggital_ch_prelu3')
        saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv4', num_filters=40, filter_size=3)),  name = 'saggital_ch_prelu4')
        saggital_ch = MaxPool2DDNNLayer(saggital_ch, name='saggital_max_pool_2', pool_size=2)
        saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv5', num_filters=60, filter_size=3)),  name = 'saggital_ch_prelu5')
        saggital_ch = DenseLayer(saggital_ch, name='saggital_d1', num_units = fc_conv)
        saggital_ch = prelu(saggital_ch, name = 'saggital_prelu_d1')
        saggital_ch = DropoutLayer(saggital_ch, name = 'saggital_l1drop', p = dropout_conv)

        # concatenate channels
        atlas_layer = InputLayer(name='in4', shape=(None, 15))
        layer = ConcatLayer(name = 'elem_channels', incomings = [axial_ch, coronal_ch, saggital_ch, atlas_layer])

        # fully connected layer
        layer = DenseLayer(layer, name='fc_2', num_units = fc_fc)
        layer = prelu(layer, name = 'prelu_f2')
        layer = DropoutLayer(layer, name = 'f2_drop', p = dropout_fc)

        # fully connected layer 
        layer = DenseLayer(layer, name='fc_3', num_units = fc_fc)
        layer = prelu(layer, name = 'prelu_f3')
        layer = DropoutLayer(layer, name = 'f3_drop', p = dropout_fc)

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
