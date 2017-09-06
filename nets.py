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


def float32(k):
        return np.cast['float32'](k)


def filters_increase_dims(l, increase_dims):
    """
    """
    in_num_filters = l.output_shape[1]
    if increase_dims:
        first_stride = (2, 2)
        out_num_filters = in_num_filters*2
    else:
        first_stride = (1, 1)
        out_num_filters = in_num_filters        

    return out_num_filters, first_stride


def projection(l_inp):
    # twice normal channels when projecting
    n_filters = l_inp.output_shape[1]*2
    l = batch_norm_dnn(Conv2DDNNLayer(l_inp, num_filters=n_filters, filter_size=1,
                                            stride=2, nonlinearity=None, pad='same',b = None))
    return l


def res_block(l_inp, name = None, increase_dim=False):
    """
    Residual Convolution block (He et al. 2015)
    # conv -> BN -> nonlin -> conv -> BN -> sum -> nonlin
    """
    
    # first figure filters/strides 
    n_filters, first_stride = filters_increase_dims(l_inp, increase_dim)

    # Convolution + batchnorm + prelu  
    l = prelu(batch_norm_dnn(Conv2DDNNLayer(l_inp, name = name+'_rl_conv1', num_filters=n_filters, filter_size=3,
                                            stride=first_stride, nonlinearity=None, pad='same',
                                            W=lasagne.init.HeNormal(gain=np.sqrt(2/(1+0.5**2)))), name = name+'_rl_batch_norm1'), name = name+'_rl_prelu1')
    # Convolution + batchnorm 
    l = batch_norm_dnn(Conv2DDNNLayer(l, name = name+'_rl_conv2', num_filters=n_filters, filter_size=3,
                                            stride=(1,1), nonlinearity=None, pad='same',
                                            W=lasagne.init.HeNormal(gain=np.sqrt(2/(1+0.5**2)))), name = name+'_rl_batch_norm2')
    if increase_dim:
        p = projection(l_inp)
    else:
        # Identity shortcut
        p = l_inp

    l = ElemwiseSumLayer([l, p], name= name+'_rl_sum')
    l = prelu(l, name = name+'_rl_batch_prelu2')
    
    return l


class AdjustVariable(object):
    """
    Handle class to update the learning rate after each iteration
    """
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None
    
    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)
        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)
        
class Rotate_batch_Iterator(BatchIterator):
    """
    handle class for on-the-fly data augmentation on batches. 
    Applying 90,180 and 270 degrees rotations and flipping
    """
    def transform(self, Xb, yb):
        Xb, yb = super(Rotate_batch_Iterator, self).transform(Xb, yb)

        # Flip a given percentage of the images at random:
        
        bs = Xb['in1'].shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)        

        x_axial = Xb['in1'][indices]
        x_cor = Xb['in2'][indices]
        x_sag = Xb['in3'][indices]

        if len(x_axial) > 0:
        
                # apply rotation to the input batch
                rotate_90 = x_axial[:,:,::-1,:].transpose(0,1,3,2)
                rotate_180 = rotate_90[:,:,::-1,:].transpose(0,1,3,2)
                #rotate_270 = rotate_180[:,:,:,::-1,:].transpose(0,1,2,4,3)

                # apply flipped versions of rotated patches
                rotate_0_flipped = x_axial[:,:,:,::-1]
                #rotate_90_flipped = rotate_90[:,:,:,:,::-1]
                rotate_180_flipped = rotate_180[:,:,:,::-1]
                #rotate_270_flipped = rotate_270[:,:,:,:,::-1]


                augmented_x = np.stack([rotate_180,
                                        rotate_0_flipped,
                                        rotate_180_flipped],
                                        axis=1)

                r_indices = np.random.randint(0,3,size=augmented_x.shape[0])

                Xb['in1'][indices] = np.stack([augmented_x[i,r_indices[i],:,:,:] for i in range(augmented_x.shape[0])])

                # apply rotation to the input batch
                rotate_90 = x_cor[:,:,::-1,:].transpose(0,1,3,2)
                rotate_180 = rotate_90[:,:,::-1,:].transpose(0,1,3,2)
                #rotate_270 = rotate_180[:,:,:,::-1,:].transpose(0,1,2,4,3)

                # apply flipped versions of rotated patches
                rotate_0_flipped = x_cor[:,:,:,::-1]
                #rotate_90_flipped = rotate_90[:,:,:,:,::-1]
                rotate_180_flipped = rotate_180[:,:,:,::-1]
                #rotate_270_flipped = rotate_270[:,:,:,:,::-1]


                augmented_x = np.stack([rotate_180,
                                        rotate_0_flipped,
                                        rotate_180_flipped],
                                        axis=1)

                r_indices = np.random.randint(0,3,size=augmented_x.shape[0])

                Xb['in2'][indices] = np.stack([augmented_x[i,r_indices[i],:,:,:] for i in range(augmented_x.shape[0])])


                # apply rotation to the input batch
                rotate_90 = x_sag[:,:,::-1,:].transpose(0,1,3,2)
                rotate_180 = rotate_90[:,:,::-1,:].transpose(0,1,3,2)
                #rotate_270 = rotate_180[:,:,:,::-1,:].transpose(0,1,2,4,3)

                # apply flipped versions of rotated patches
                rotate_0_flipped = x_sag[:,:,:,::-1]
                #rotate_90_flipped = rotate_90[:,:,:,:,::-1]
                rotate_180_flipped = rotate_180[:,:,:,::-1]
                #rotate_270_flipped = rotate_270[:,:,:,:,::-1]


                augmented_x = np.stack([rotate_180,
                                        rotate_0_flipped,
                                        rotate_180_flipped],
                                        axis=1)

                r_indices = np.random.randint(0,3,size=augmented_x.shape[0])

                Xb['in3'][indices] = np.stack([augmented_x[i,r_indices[i],:,:,:] for i in range(augmented_x.shape[0])])
                
        return Xb, yb




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


    if (options['experiment'] == 'R1CONV_180_555_195_195_d_0.7_adam_N4'  ) or (options['experiment'] == 'CONV_180_555_195_195_d_0.7_adam_N4') or (options['experiment'] == 'R2CONV_180_555_195_195_d_0.7_adam_N4')  or (options['experiment'] == 'CONV_rand2_180_555_195_195_d_0.7_adam_N4')   or (options['experiment'] == 'CONV_rand3_180_555_195_195_d_0.7_adam_N4') or (options['experiment'] == 'CONV_rand3_180_555_195_195_d_0.7_adam_RE_N4') :
        fc_conv = 180
        fc_fc = fc_conv + 15
        dropout_conv = 0.7
        dropout_fc = 0.7
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
            update = updates.adam,
            update_learning_rate=0.0001,   
            on_epoch_finished=[
                save_weights,
                save_training_history,
                early_stopping,
            ],
            verbose= t_verbose,
            max_epochs= num_epochs,
            train_split=TrainSplit(eval_size= train_split_perc),
        )

    if (options['experiment'] == 'CONV_120_360_135_135_d_0.4_adam_N4'):
    
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
        layer = ConcatLayer(name = 'elem_channels', incomings = [axial_ch, coronal_ch, saggital_ch])

        # fully connected layer
        layer = DenseLayer(layer, name='fc_2', num_units = fc_fc)
        layer = prelu(layer, name = 'prelu_f2')
        layer = DropoutLayer(layer, name = 'f2_drop', p = dropout_fc)

        atlas_layer = InputLayer(name='in4', shape=(None, 15))
        layer = ConcatLayer(name = 'elem_channels_2', incomings = [layer, atlas_layer])

        # fully connected layer 
        layer = DenseLayer(layer, name='fc_3', num_units = fc_fc+15)
        layer = prelu(layer, name = 'prelu_f3')
        layer = DropoutLayer(layer, name = 'f3_drop', p = dropout_fc)

        # softmax
        net_layer = DenseLayer(layer, name='out_layer', num_units = 15, nonlinearity=softmax)

        net =  NeuralNet(
            layers= net_layer,
            objective_loss_function=objectives.categorical_crossentropy,
            update = updates.adam,
            update_learning_rate=0.001,   
            on_epoch_finished=[
                save_weights,
                save_training_history,
                early_stopping,
            ],
            verbose= t_verbose,
            max_epochs= num_epochs,
            train_split=TrainSplit(eval_size= train_split_perc),
        )
    if (options['experiment'] == 'CONV_180_555_555_555_d_0.7_adam_N4'):
    
        fc_conv = 180
        fc_fc = 555
        dropout_conv = 0.7
        dropout_fc = 0.7
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
            update = updates.adam,
            update_learning_rate=0.001,   
            on_epoch_finished=[
                save_weights,
                save_training_history,
                early_stopping,
            ],
                verbose= t_verbose,
                max_epochs= num_epochs,
                train_split=TrainSplit(eval_size= train_split_perc),
        )

    if (options['experiment'] == 'CONV_180_540_270_N4') or (options['experiment'] == 'CONV_' + str(ps) + '_180_540_270_N4' ) or (options['experiment'] == 'CONV_180_540_270_NOBORDER_N4')  or (options['experiment'] == 'CONV_180_540_270_K3_N4') or (options['experiment'] == 'CONV_180_540_270'):
        # CONV_180_540_270_NOBORDER_N4 (no border sampling)
        fc_conv = 180
        fc_fc = 180 
        dropout_conv = 0.5
        dropout_fc = 0.5
        
        # --------------------------------------------------
        # channel_1: axial
        # --------------------------------------------------
        # input: 32

        axial_ch = InputLayer(name='in1', shape=(None, num_channels, ps, ps))
        #axial_ch = DropoutLayer(InputLayer(name='in1', shape=(None, num_channels, ps, ps)), name ='Dropout_in_2', p =.2)
        axial_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(axial_ch, name='axial_ch_conv1', num_filters=20, filter_size=3)),  name = 'axial_ch_prelu1')
        axial_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(axial_ch, name='axial_ch_conv2', num_filters=20, filter_size=3)),  name = 'axial_ch_prelu2')
        axial_ch = MaxPool2DDNNLayer(axial_ch, name='axial_max_pool_1', pool_size=2)
        axial_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(axial_ch, name='axial_ch_conv3', num_filters=40, filter_size=3)),  name = 'axial_ch_prelu3')
        axial_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(axial_ch, name='axial_ch_conv4', num_filters=40, filter_size=3)),  name = 'axial_ch_prelu4')
        axial_ch = MaxPool2DDNNLayer(axial_ch, name='axial_max_pool_2', pool_size=2)
        axial_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(axial_ch, name='axial_ch_conv5', num_filters=60, filter_size=3)),  name = 'axial_ch_prelu5')
        axial_ch = DropoutLayer(axial_ch, name = 'axial_l1drop', p = dropout_conv)
        axial_ch = DenseLayer(axial_ch, name='axial_d1', num_units = fc_conv)
        axial_ch = prelu(axial_ch, name = 'axial_prelu_d1')
        # --------------------------------------------------
        # channel_1: coronal
        # --------------------------------------------------
        # input: 32
        #coronal_ch = DropoutLayer(InputLayer(name='in2', shape=(None, num_channels, ps, ps)), name ='Dropout_in_2', p =.2)
        coronal_ch = InputLayer(name='in2', shape=(None, num_channels, ps, ps))
        coronal_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(coronal_ch, name='coronal_ch_conv1', num_filters=20, filter_size=3)),  name = 'coronal_ch_prelu1')
        coronal_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(coronal_ch, name='coronal_ch_conv2', num_filters=20, filter_size=3)),  name = 'coronal_ch_prelu2')
        coronal_ch = MaxPool2DDNNLayer(coronal_ch, name='coronal_max_pool_1', pool_size=2)
        coronal_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(coronal_ch, name='coronal_ch_conv3', num_filters=40, filter_size=3)),  name = 'coronal_ch_prelu3')
        coronal_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(coronal_ch, name='coronal_ch_conv4', num_filters=40, filter_size=3)),  name = 'coronal_ch_prelu4')
        coronal_ch = MaxPool2DDNNLayer(coronal_ch, name='coronal_max_pool_2', pool_size=2)
        coronal_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(coronal_ch, name='coronal_ch_conv5', num_filters=60, filter_size=3)),  name = 'coronal_ch_prelu5')
        coronal_ch = DropoutLayer(coronal_ch, name = 'coronal_l1drop', p = dropout_conv)
        coronal_ch = DenseLayer(coronal_ch, name='coronal_d1', num_units = fc_conv)
        coronal_ch = prelu(coronal_ch, name = 'coronal_prelu_d1')
        # --------------------------------------------------
        # channel_1: saggital
        # --------------------------------------------------
        #input: 32
        #saggital_ch = DropoutLayer(InputLayer(name='in3', shape=(None, num_channels, ps, ps)), name ='Dropout_in_3', p =.2)
        saggital_ch = InputLayer(name='in3', shape=(None, num_channels, ps, ps))
        saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv1', num_filters=20, filter_size=3)),  name = 'saggital_ch_prelu1')
        saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv2', num_filters=20, filter_size=3)),  name = 'saggital_ch_prelu2')
        saggital_ch = MaxPool2DDNNLayer(saggital_ch, name='saggital_max_pool_1', pool_size=2)
        saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv3', num_filters=40, filter_size=3)),  name = 'saggital_ch_prelu3')
        saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv4', num_filters=40, filter_size=3)),  name = 'saggital_ch_prelu4')
        saggital_ch = MaxPool2DDNNLayer(saggital_ch, name='saggital_max_pool_2', pool_size=2)
        saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv5', num_filters=60, filter_size=3)),  name = 'saggital_ch_prelu5')
        saggital_ch = DropoutLayer(saggital_ch, name = 'saggital_l1drop', p = dropout_conv)
        saggital_ch = DenseLayer(saggital_ch, name='saggital_d1', num_units = fc_conv)
        saggital_ch = prelu(saggital_ch, name = 'saggital_prelu_d1')
        
        # FC
        layer = ConcatLayer(name = 'elem_channels', incomings = [axial_ch, coronal_ch, saggital_ch])
        layer = DropoutLayer(layer, name = 'f1_drop', p = dropout_fc)        
        layer = DenseLayer(layer, name='FC1', num_units =540)
        layer = prelu(layer, name = 'prelu_f1')
        layer = DropoutLayer(layer, name = 'f2_drop', p = dropout_fc)        
        
        # concatenate channels
        atlas_layer = DropoutLayer(InputLayer(name='in4', shape=(None, 15)), name = 'Dropout_atlas', p = .2)
        atlas_layer = InputLayer(name='in4', shape=(None, 15))
        layer = ConcatLayer(name = 'elem_channels2', incomings = [layer, atlas_layer])

        # fully connected layer
        layer = DenseLayer(layer, name='fc_2', num_units = 270)
        layer = prelu(layer, name = 'prelu_f2')
        
        # softmax
        net_layer = DenseLayer(layer, name='out_layer', num_units = 15, nonlinearity=softmax)

        net =  NeuralNet(
            layers= net_layer,
            objective_loss_function=objectives.categorical_crossentropy,
            update = updates.adam,
            update_learning_rate=0.001,   
            on_epoch_finished=[
                save_weights,
                save_training_history,
                early_stopping,
            ],
            verbose= t_verbose,
            max_epochs= num_epochs,
            train_split=TrainSplit(eval_size= train_split_perc),
        )

    if (options['experiment'] == 'CONVv2_180_540_270_N4') or (options['experiment'] == 'CONV_180_540_270_ae2_N4'):
    
        fc_conv = 180
        fc_fc = 180 
        dropout_conv = 0.5
        dropout_fc = 0.5
        # --------------------------------------------------
        # channel_1: axial
        # --------------------------------------------------
        # input: 32

        axial_ch = InputLayer(name='in1', shape=(None, num_channels, ps, ps))
        #axial_ch = DropoutLayer(InputLayer(name='in1', shape=(None, num_channels, ps, ps)), name ='Dropout_in_2', p =.2)
        axial_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(axial_ch, name='axial_ch_conv1', num_filters=20, filter_size=3)),  name = 'axial_ch_prelu1')
        axial_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(axial_ch, name='axial_ch_conv2', num_filters=20, filter_size=3)),  name = 'axial_ch_prelu2')
        axial_ch = MaxPool2DDNNLayer(axial_ch, name='axial_max_pool_1', pool_size=2)
        axial_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(axial_ch, name='axial_ch_conv3', num_filters=40, filter_size=3)),  name = 'axial_ch_prelu3')
        axial_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(axial_ch, name='axial_ch_conv4', num_filters=40, filter_size=3)),  name = 'axial_ch_prelu4')
        axial_ch = MaxPool2DDNNLayer(axial_ch, name='axial_max_pool_2', pool_size=2)
        axial_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(axial_ch, name='axial_ch_conv5', num_filters=60, filter_size=3)),  name = 'axial_ch_prelu5')
        axial_ch = DropoutLayer(axial_ch, name = 'axial_l1drop', p = dropout_conv)
        axial_ch = DenseLayer(axial_ch, name='axial_d1', num_units = fc_conv)
        axial_ch = prelu(axial_ch, name = 'axial_prelu_d1')
        # --------------------------------------------------
        # channel_1: coronal
        # --------------------------------------------------
        # input: 32
        #coronal_ch = DropoutLayer(InputLayer(name='in2', shape=(None, num_channels, ps, ps)), name ='Dropout_in_2', p =.2)
        coronal_ch = InputLayer(name='in2', shape=(None, num_channels, ps, ps))
        coronal_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(coronal_ch, name='coronal_ch_conv1', num_filters=20, filter_size=3)),  name = 'coronal_ch_prelu1')
        coronal_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(coronal_ch, name='coronal_ch_conv2', num_filters=20, filter_size=3)),  name = 'coronal_ch_prelu2')
        coronal_ch = MaxPool2DDNNLayer(coronal_ch, name='coronal_max_pool_1', pool_size=2)
        coronal_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(coronal_ch, name='coronal_ch_conv3', num_filters=40, filter_size=3)),  name = 'coronal_ch_prelu3')
        coronal_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(coronal_ch, name='coronal_ch_conv4', num_filters=40, filter_size=3)),  name = 'coronal_ch_prelu4')
        coronal_ch = MaxPool2DDNNLayer(coronal_ch, name='coronal_max_pool_2', pool_size=2)
        coronal_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(coronal_ch, name='coronal_ch_conv5', num_filters=60, filter_size=3)),  name = 'coronal_ch_prelu5')
        coronal_ch = DropoutLayer(coronal_ch, name = 'coronal_l1drop', p = dropout_conv)
        coronal_ch = DenseLayer(coronal_ch, name='coronal_d1', num_units = fc_conv)
        coronal_ch = prelu(coronal_ch, name = 'coronal_prelu_d1')
        # --------------------------------------------------
        # channel_1: saggital
        # --------------------------------------------------
        #input: 32
        #saggital_ch = DropoutLayer(InputLayer(name='in3', shape=(None, num_channels, ps, ps)), name ='Dropout_in_3', p =.2)
        saggital_ch = InputLayer(name='in3', shape=(None, num_channels, ps, ps))
        saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv1', num_filters=20, filter_size=3)),  name = 'saggital_ch_prelu1')
        saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv2', num_filters=20, filter_size=3)),  name = 'saggital_ch_prelu2')
        saggital_ch = MaxPool2DDNNLayer(saggital_ch, name='saggital_max_pool_1', pool_size=2)
        saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv3', num_filters=40, filter_size=3)),  name = 'saggital_ch_prelu3')
        saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv4', num_filters=40, filter_size=3)),  name = 'saggital_ch_prelu4')
        saggital_ch = MaxPool2DDNNLayer(saggital_ch, name='saggital_max_pool_2', pool_size=2)
        saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv5', num_filters=60, filter_size=3)),  name = 'saggital_ch_prelu5')
        saggital_ch = DropoutLayer(saggital_ch, name = 'saggital_l1drop', p = dropout_conv)
        saggital_ch = DenseLayer(saggital_ch, name='saggital_d1', num_units = fc_conv)
        saggital_ch = prelu(saggital_ch, name = 'saggital_prelu_d1')
        
        # FC
        atlas_layer = InputLayer(name='in4', shape=(None, 15))
        layer = ConcatLayer(name = 'elem_channels', incomings = [axial_ch, coronal_ch, saggital_ch])
        layer = DropoutLayer(layer, name = 'f1_drop', p = dropout_fc)
        layer = DenseLayer(layer, name='FC1', num_units =540)
        layer = prelu(layer, name = 'prelu_f1')
        layer = DropoutLayer(layer, name = 'f2_drop', p = dropout_fc)        

        layer = ConcatLayer(name = 'elem_channels2', incomings = [layer, atlas_layer])

        # fully connected layer
        layer = DenseLayer(layer, name='fc_3', num_units = 270)
        layer = prelu(layer, name = 'prelu_f2')
        
        # softmax
        net_layer = DenseLayer(layer, name='out_layer', num_units = 15, nonlinearity=softmax)

        net =  NeuralNet(
            layers= net_layer,
            objective_loss_function=objectives.categorical_crossentropy,
            update = updates.adam,
            update_learning_rate=0.001,   
            on_epoch_finished=[
                save_weights,
                save_training_history,
                early_stopping,
            ],
            verbose= t_verbose,
            max_epochs= num_epochs,
            train_split=TrainSplit(eval_size= train_split_perc),
        )

    if (options['experiment'] == 'CONV_re3_180_540_270_RE_N4') or (options['experiment'] == 'CONV_re4_180_540_270_RE_N4'):
    
        fc_conv = 180
        fc_fc = 180 
        dropout_conv = 0.5
        dropout_fc = 0.5
        # --------------------------------------------------
        # channel_1: axial
        # --------------------------------------------------
        # input: 32

        axial_ch = InputLayer(name='in1', shape=(None, num_channels, ps, ps))
        #axial_ch = DropoutLayer(InputLayer(name='in1', shape=(None, num_channels, ps, ps)), name ='Dropout_in_2', p =.2)
        axial_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(axial_ch, name='axial_ch_conv1', num_filters=20, filter_size=3)),  name = 'axial_ch_prelu1')
        axial_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(axial_ch, name='axial_ch_conv2', num_filters=20, filter_size=3)),  name = 'axial_ch_prelu2')
        axial_ch = MaxPool2DDNNLayer(axial_ch, name='axial_max_pool_1', pool_size=2)
        axial_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(axial_ch, name='axial_ch_conv3', num_filters=40, filter_size=3)),  name = 'axial_ch_prelu3')
        axial_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(axial_ch, name='axial_ch_conv4', num_filters=40, filter_size=3)),  name = 'axial_ch_prelu4')
        axial_ch = MaxPool2DDNNLayer(axial_ch, name='axial_max_pool_2', pool_size=2)
        axial_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(axial_ch, name='axial_ch_conv5', num_filters=60, filter_size=3)),  name = 'axial_ch_prelu5')
        axial_ch = DropoutLayer(axial_ch, name = 'axial_l1drop', p = dropout_conv)
        axial_ch = DenseLayer(axial_ch, name='axial_d1', num_units = fc_conv)
        axial_ch = prelu(axial_ch, name = 'axial_prelu_d1')
        # --------------------------------------------------
        # channel_1: coronal
        # --------------------------------------------------
        # input: 32
        #coronal_ch = DropoutLayer(InputLayer(name='in2', shape=(None, num_channels, ps, ps)), name ='Dropout_in_2', p =.2)
        coronal_ch = InputLayer(name='in2', shape=(None, num_channels, ps, ps))
        coronal_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(coronal_ch, name='coronal_ch_conv1', num_filters=20, filter_size=3)),  name = 'coronal_ch_prelu1')
        coronal_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(coronal_ch, name='coronal_ch_conv2', num_filters=20, filter_size=3)),  name = 'coronal_ch_prelu2')
        coronal_ch = MaxPool2DDNNLayer(coronal_ch, name='coronal_max_pool_1', pool_size=2)
        coronal_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(coronal_ch, name='coronal_ch_conv3', num_filters=40, filter_size=3)),  name = 'coronal_ch_prelu3')
        coronal_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(coronal_ch, name='coronal_ch_conv4', num_filters=40, filter_size=3)),  name = 'coronal_ch_prelu4')
        coronal_ch = MaxPool2DDNNLayer(coronal_ch, name='coronal_max_pool_2', pool_size=2)
        coronal_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(coronal_ch, name='coronal_ch_conv5', num_filters=60, filter_size=3)),  name = 'coronal_ch_prelu5')
        coronal_ch = DropoutLayer(coronal_ch, name = 'coronal_l1drop', p = dropout_conv)
        coronal_ch = DenseLayer(coronal_ch, name='coronal_d1', num_units = fc_conv)
        coronal_ch = prelu(coronal_ch, name = 'coronal_prelu_d1')
        # --------------------------------------------------
        # channel_1: saggital
        # --------------------------------------------------
        #input: 32
        #saggital_ch = DropoutLayer(InputLayer(name='in3', shape=(None, num_channels, ps, ps)), name ='Dropout_in_3', p =.2)
        saggital_ch = InputLayer(name='in3', shape=(None, num_channels, ps, ps))
        saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv1', num_filters=20, filter_size=3)),  name = 'saggital_ch_prelu1')
        saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv2', num_filters=20, filter_size=3)),  name = 'saggital_ch_prelu2')
        saggital_ch = MaxPool2DDNNLayer(saggital_ch, name='saggital_max_pool_1', pool_size=2)
        saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv3', num_filters=40, filter_size=3)),  name = 'saggital_ch_prelu3')
        saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv4', num_filters=40, filter_size=3)),  name = 'saggital_ch_prelu4')
        saggital_ch = MaxPool2DDNNLayer(saggital_ch, name='saggital_max_pool_2', pool_size=2)
        saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv5', num_filters=60, filter_size=3)),  name = 'saggital_ch_prelu5')
        saggital_ch = DropoutLayer(saggital_ch, name = 'saggital_l1drop', p = dropout_conv)
        saggital_ch = DenseLayer(saggital_ch, name='saggital_d1', num_units = fc_conv)
        saggital_ch = prelu(saggital_ch, name = 'saggital_prelu_d1')
        
        # FC

        layer = ConcatLayer(name = 'elem_channels', incomings = [axial_ch, coronal_ch, saggital_ch])
        layer = DropoutLayer(layer, name = 'f1_drop', p = dropout_fc)

        layer = DenseLayer(layer, name='FC1', num_units =540)
        layer = prelu(layer, name = 'prelu_f1')

        atlas_layer = InputLayer(name='in4', shape=(None, 15))
        layer = ConcatLayer(name = 'elem_channels2', incomings = [layer, atlas_layer])

        # fully connected layer
        layer = DenseLayer(layer, name='fc_3', num_units = 270)
        layer = prelu(layer, name = 'prelu_f2')
        layer = DropoutLayer(layer, name = 'f2_drop', p = dropout_fc)
                
        # softmax
        net_layer = DenseLayer(layer, name='out_layer', num_units = 15, nonlinearity=softmax)

        net =  NeuralNet(
            layers= net_layer,
            objective_loss_function=objectives.categorical_crossentropy,
            update = updates.adam,
            update_learning_rate=0.001,   
            on_epoch_finished=[
                save_weights, 
                save_training_history,
                early_stopping,
            ],
            verbose= t_verbose,
            max_epochs= num_epochs,
            train_split=TrainSplit(eval_size= train_split_perc),
        )

        
    if (options['experiment'] == 'CONV32_res2_160_480_240_N4'):
    
        fc_conv = 160
        fc_fc = 160
        dropout_conv = 0.5
        dropout_fc = 0.5
        init_num_filters = 20
        # --------------------------------------------------
        # channel_1: axial
        # --------------------------------------------------
        # input: 32

        axial_ch = InputLayer(name='in1', shape=(None, num_channels, ps, ps))
        axial_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(axial_ch, name='axial_cnn_0', num_filters=init_num_filters, filter_size=3, pad = 'same')),  name = 'axial_ch_prelu0')
        axial_ch = res_block(axial_ch, name = 'axial_cnn_1', increase_dim=False)
        axial_ch = res_block(axial_ch, name = 'axial_cnn_2', increase_dim=False)
        axial_ch = res_block(axial_ch, name = 'axial_cnn_3', increase_dim=True)
        axial_ch = res_block(axial_ch, name = 'axial_cnn_4', increase_dim=False)
        axial_ch = res_block(axial_ch, name = 'axial_cnn_5', increase_dim=True)
        axial_ch = res_block(axial_ch, name = 'axial_cnn_6', increase_dim=False)
        axial_ch = res_block(axial_ch, name = 'axial_cnn_7', increase_dim=True)
    
        axial_ch = DropoutLayer(axial_ch, name = 'axial_l1drop', p = dropout_conv)
        axial_ch = DenseLayer(axial_ch, name='axial_d1', num_units = fc_conv)
        axial_ch = prelu(axial_ch, name = 'axial_prelu_d1')

        coronal_ch = InputLayer(name='in2', shape=(None, num_channels, ps, ps))
        coronal_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(coronal_ch, name='coronal_cnn_0', num_filters=init_num_filters, filter_size=3, pad = 'same')),  name = 'coronal_ch_prelu0')
        coronal_ch = res_block(coronal_ch, name = 'coronal_cnn_1', increase_dim=False)
        coronal_ch = res_block(coronal_ch, name = 'coronal_cnn_2', increase_dim=False)
        coronal_ch = res_block(coronal_ch, name = 'coronal_cnn_3', increase_dim=True)
        coronal_ch = res_block(coronal_ch, name = 'coronal_cnn_4', increase_dim=False)
        coronal_ch = res_block(coronal_ch, name = 'coronal_cnn_5', increase_dim=True)
        coronal_ch = res_block(coronal_ch, name = 'coronal_cnn_6', increase_dim=False)
        coronal_ch = res_block(coronal_ch, name = 'coronal_cnn_7', increase_dim=True)

        coronal_ch = DropoutLayer(coronal_ch, name = 'coronal_l1drop', p = dropout_conv)
        coronal_ch = DenseLayer(coronal_ch, name='coronal_d1', num_units = fc_conv)
        coronal_ch = prelu(coronal_ch, name = 'coronal_prelu_d1')

        saggital_ch = InputLayer(name='in3', shape=(None, num_channels, ps, ps))
        saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_cnn_0', num_filters=init_num_filters, filter_size=3, pad = 'same')),  name = 'saggital_ch_prelu0')
        saggital_ch = res_block(saggital_ch, name = 'saggital_cnn_1', increase_dim=False)
        saggital_ch = res_block(saggital_ch, name = 'saggital_cnn_2', increase_dim=False)
        saggital_ch = res_block(saggital_ch, name = 'saggital_cnn_3', increase_dim=True)
        saggital_ch = res_block(saggital_ch, name = 'saggital_cnn_4', increase_dim=False)
        saggital_ch = res_block(saggital_ch, name = 'saggital_cnn_5', increase_dim=True)
        saggital_ch = res_block(saggital_ch, name = 'saggital_cnn_6', increase_dim=False)
        saggital_ch = res_block(saggital_ch, name = 'saggital_cnn_7', increase_dim=True)

        saggital_ch = DropoutLayer(saggital_ch, name = 'saggital_l1drop', p = dropout_conv)
        saggital_ch = DenseLayer(saggital_ch, name='saggital_d1', num_units = fc_conv)
        saggital_ch = prelu(saggital_ch, name = 'saggital_prelu_d1')

        # FC  
        layer = ConcatLayer(name = 'elem_channels', incomings = [axial_ch, coronal_ch, saggital_ch])
        layer = DropoutLayer(layer, name = 'f1_drop', p = dropout_fc)

        layer = DenseLayer(layer, name='FC1', num_units =480)
        layer = prelu(layer, name = 'prelu_f1')
        layer = DropoutLayer(layer, name = 'f2_drop', p = dropout_fc)
        
        atlas_layer = InputLayer(name='in4', shape=(None, 15))
        layer = ConcatLayer(name = 'elem_channels2', incomings = [layer, atlas_layer])
        
        # fully connected layer
        layer = DenseLayer(layer, name='fc_2', num_units = 240)
        layer = prelu(layer, name = 'prelu_f2')
        
        # softmax
        net_layer = DenseLayer(layer, name='out_layer', num_units = 15, nonlinearity=softmax)

        net =  NeuralNet(
            layers= net_layer,
            objective_loss_function=objectives.categorical_crossentropy,
            update = updates.adam,
            update_learning_rate=0.001,   
            on_epoch_finished=[
                save_weights,
                save_training_history,
                early_stopping,
            ],
            verbose= t_verbose,
            max_epochs= num_epochs,
            train_split=TrainSplit(eval_size= train_split_perc),
        )


    if (options['experiment'] == 'CONV_180_540_270_DA_N4'):
    
        fc_conv = 180
        fc_fc = 180 
        dropout_conv = 0.5
        dropout_fc = 0.5

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

        # --------------------------------------------------
        # channel_2: coronal
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

        coronal_ch = DropoutLayer(coronal_ch, name = 'coronal_l1drop', p = dropout_conv)
        coronal_ch = DenseLayer(coronal_ch, name='coronal_d1', num_units = fc_conv)
        coronal_ch = prelu(coronal_ch, name = 'coronal_prelu_d1')

        # --------------------------------------------------
        # channel_3: saggital
        # --------------------------------------------------
        # input: 32

        saggital_ch = InputLayer(name='in3', shape=(None, num_channels, ps, ps))
        
        saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv1', num_filters=20, filter_size=3)),  name = 'saggital_ch_prelu1')
        saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv2', num_filters=20, filter_size=3)),  name = 'saggital_ch_prelu2')
        saggital_ch = MaxPool2DDNNLayer(saggital_ch, name='saggital_max_pool_1', pool_size=2)
        
        saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv3', num_filters=40, filter_size=3)),  name = 'saggital_ch_prelu3')
        saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv4', num_filters=40, filter_size=3)),  name = 'saggital_ch_prelu4')
        saggital_ch = MaxPool2DDNNLayer(saggital_ch, name='saggital_max_pool_2', pool_size=2)

        saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv5', num_filters=60, filter_size=3)),  name = 'saggital_ch_prelu5')

        saggital_ch = DropoutLayer(saggital_ch, name = 'saggital_l1drop', p = dropout_conv)
        saggital_ch = DenseLayer(saggital_ch, name='saggital_d1', num_units = fc_conv)
        saggital_ch = prelu(saggital_ch, name = 'saggital_prelu_d1')

    
        # FC 
        layer = ConcatLayer(name = 'elem_channels', incomings = [axial_ch, coronal_ch, saggital_ch])


        layer = DropoutLayer(layer, name = 'f1_drop', p = dropout_fc)        
        layer = DenseLayer(layer, name='FC1', num_units =540)
        layer = prelu(layer, name = 'prelu_f1')
        layer = DropoutLayer(layer, name = 'f2_drop', p = dropout_fc)        

        
        #layer = DenseLayer(layer, name='FC1_2', num_units =540)
        #layer = prelu(layer, name = 'prelu_f1_2')
        #layer = DropoutLayer(layer, name = 'f1_2_drop', p = dropout_fc)        
        
        # concatenate channels
        #atlas_layer = DropoutLayer(InputLayer(name='in4', shape=(None, 15)), name = 'Dropout_atlas', p = .1)
        atlas_layer = InputLayer(name='in4', shape=(None, 15))
        layer = ConcatLayer(name = 'elem_channels2', incomings = [layer, atlas_layer])

        # fully connected layer
        layer = DenseLayer(layer, name='fc_2', num_units = 270)
        layer = prelu(layer, name = 'prelu_f2')
        
        # softmax
        net_layer = DenseLayer(layer, name='out_layer', num_units = 15, nonlinearity=softmax)

        net =  NeuralNet(
            layers= net_layer,
            objective_loss_function=objectives.categorical_crossentropy,
            batch_iterator_train=Rotate_batch_Iterator(batch_size=128),
            update = updates.adam,
            update_learning_rate=0.001,   
            on_epoch_finished=[
                save_weights,
                save_training_history,
                early_stopping,
            ],
            verbose= t_verbose,
            max_epochs= num_epochs,
            train_split=TrainSplit(eval_size= train_split_perc),
        )

    if (options['experiment'] == 'CONV_180_540_270_NOATLAS_N4') or (options['experiment'] == 'CONV_180_540_270_NOATLAS'):
    
        fc_conv = 180
        fc_fc = 180 
        dropout_conv = 0.5
        dropout_fc = 0.5

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

        # --------------------------------------------------
        # channel_2: coronal
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

        coronal_ch = DropoutLayer(coronal_ch, name = 'coronal_l1drop', p = dropout_conv)
        coronal_ch = DenseLayer(coronal_ch, name='coronal_d1', num_units = fc_conv)
        coronal_ch = prelu(coronal_ch, name = 'coronal_prelu_d1')

        # --------------------------------------------------
        # channel_3: saggital
        # --------------------------------------------------
        # input: 32

        saggital_ch = InputLayer(name='in3', shape=(None, num_channels, ps, ps))
        
        saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv1', num_filters=20, filter_size=3)),  name = 'saggital_ch_prelu1')
        saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv2', num_filters=20, filter_size=3)),  name = 'saggital_ch_prelu2')
        saggital_ch = MaxPool2DDNNLayer(saggital_ch, name='saggital_max_pool_1', pool_size=2)
        
        saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv3', num_filters=40, filter_size=3)),  name = 'saggital_ch_prelu3')
        saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv4', num_filters=40, filter_size=3)),  name = 'saggital_ch_prelu4')
        saggital_ch = MaxPool2DDNNLayer(saggital_ch, name='saggital_max_pool_2', pool_size=2)

        saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv5', num_filters=60, filter_size=3)),  name = 'saggital_ch_prelu5')

        saggital_ch = DropoutLayer(saggital_ch, name = 'saggital_l1drop', p = dropout_conv)
        saggital_ch = DenseLayer(saggital_ch, name='saggital_d1', num_units = fc_conv)
        saggital_ch = prelu(saggital_ch, name = 'saggital_prelu_d1')

    
        # FC 
        layer = ConcatLayer(name = 'elem_channels', incomings = [axial_ch, coronal_ch, saggital_ch])


        layer = DropoutLayer(layer, name = 'f1_drop', p = dropout_fc)        
        layer = DenseLayer(layer, name='FC1', num_units =540)
        layer = prelu(layer, name = 'prelu_f1')
        layer = DropoutLayer(layer, name = 'f2_drop', p = dropout_fc)        

        
        #layer = DenseLayer(layer, name='FC1_2', num_units =540)
        #layer = prelu(layer, name = 'prelu_f1_2')
        #layer = DropoutLayer(layer, name = 'f1_2_drop', p = dropout_fc)        
        
        # concatenate channels
        #atlas_layer = DropoutLayer(InputLayer(name='in4', shape=(None, 15)), name = 'Dropout_atlas', p = .1)
        #atlas_layer = InputLayer(name='in4', shape=(None, 15))
        #layer = ConcatLayer(name = 'elem_channels2', incomings = [layer, atlas_layer])

        # fully connected layer
        layer = DenseLayer(layer, name='fc_2', num_units = 270)
        layer = prelu(layer, name = 'prelu_f2')
        
        # softmax
        net_layer = DenseLayer(layer, name='out_layer', num_units = 15, nonlinearity=softmax)

        net =  NeuralNet(
            layers= net_layer,
            objective_loss_function=objectives.categorical_crossentropy,
            #batch_iterator_train=Rotate_batch_Iterator(batch_size=128),
            update = updates.adam,
            update_learning_rate=0.001,   
            on_epoch_finished=[
                save_weights,
                save_training_history,
                early_stopping,
            ],
            verbose= t_verbose,
            max_epochs= num_epochs,
            train_split=TrainSplit(eval_size= train_split_perc),
        )


    if (options['experiment'] == 'CONV25_256_768_384_N4'):
    
        fc_conv = 256
        fc_fc = 768 
        dropout_conv = 0.5
        dropout_fc = 0.5

        # --------------------------------------------------
        # channel_1: axial
        # --------------------------------------------------
        # input: 32

        a_ch_input = InputLayer(name='in1', shape=(None, num_channels, ps, ps))

        num_filters = 32
        a_1_conv_1 = Conv2DDNNLayer(a_ch_input, name='a_1_conv_1', num_filters=num_filters, filter_size=3)
        a_1_bn_1 = batch_norm_dnn(a_1_conv_1)
        a_1_nl_1 = prelu(a_1_bn_1,  name = 'a_1_nl_1')
        a_1_conv_2 = Conv2DDNNLayer(a_1_nl_1, name='a_1_conv_2', num_filters=num_filters, filter_size=3)
        a_1_bn_2 = batch_norm_dnn(a_1_conv_2)
        a_1_nl_2 = prelu(a_1_bn_2,  name = 'a_1_nl_2')
        a_1_conv_3 = Conv2DDNNLayer(a_1_nl_2, name='a_1_conv_3', num_filters=num_filters, filter_size=3)
        a_1_bn_3 = batch_norm_dnn(a_1_conv_3)
        a_1_nl_3 = prelu(a_1_bn_3, name = 'a_1_l_3')

        num_filters = 64
        a_2_conv_1 = Conv2DDNNLayer(a_1_nl_3, name='a_2_conv_1', num_filters=num_filters, filter_size=3)
        a_2_bn_1 = batch_norm_dnn(a_2_conv_1)
        a_2_nl_1 = prelu(a_2_bn_1,  name = 'a_2_nl_1')
        a_2_conv_2 = Conv2DDNNLayer(a_2_nl_1, name='a_2_conv_2', num_filters=num_filters, filter_size=3)
        a_2_bn_2 = batch_norm_dnn(a_2_conv_2)
        a_2_nl_2 = prelu(a_2_bn_2,  name = 'a_2_nl_2')
        a_2_conv_3 = Conv2DDNNLayer(a_2_nl_2, name='a_2_conv_3', num_filters=num_filters, filter_size=3)
        a_2_bn_3 = batch_norm_dnn(a_2_conv_3)
        a_2_nl_3 = prelu(a_2_bn_3,  name = 'a_2_nl_3')
        
        num_filters = 128
        a_3_conv_1 = Conv2DDNNLayer(a_2_nl_3, name='a_3_conv_1', num_filters=num_filters, filter_size=3)
        a_3_bn_1 = batch_norm_dnn(a_3_conv_1)
        a_3_nl_1 = prelu(a_3_bn_1,  name = 'a_3_nl_1')
        a_3_conv_2 = Conv2DDNNLayer(a_3_nl_1, name='a_3_conv_2', num_filters=num_filters, filter_size=3)
        a_3_bn_2 = batch_norm_dnn(a_3_conv_2)
        a_3_nl_2 = prelu(a_3_bn_2,  name = 'a_3_nl_2')
        a_3_conv_3 = Conv2DDNNLayer(a_3_nl_2, name='a_3_conv_3', num_filters=num_filters, filter_size=3)
        a_3_bn_3 = batch_norm_dnn(a_3_conv_3)
        a_3_nl_3 = prelu(a_3_bn_3,  name = 'a_3_nl_3')

        a_4_fc = DenseLayer(a_3_nl_3, name='a_d1', num_units = fc_conv)
        a_out = prelu(a_4_fc, name = 'a_fc_d1')



        # --------------------------------------------------
        # channel_1: coronal
        # --------------------------------------------------
        # input: 32

        cor_ch_input = InputLayer(name='in2', shape=(None, num_channels, ps, ps))

        num_filters = 32
        cor_1_conv_1 = Conv2DDNNLayer(cor_ch_input, name='cor_1_conv_1', num_filters=num_filters, filter_size=3, W=a_1_conv_1.W)
        cor_1_bn_1 = batch_norm_dnn(cor_1_conv_1)
        cor_1_nl_1 = prelu(cor_1_bn_1,  name = 'cor_1_nl_1')
        cor_1_conv_2 = Conv2DDNNLayer(cor_1_nl_1, name='cor_1_conv_2', num_filters=num_filters, filter_size=3, W=a_1_conv_2.W)
        cor_1_bn_2 = batch_norm_dnn(cor_1_conv_2)
        cor_1_nl_2 = prelu(cor_1_bn_2,  name = 'cor_1_nl_2')
        cor_1_conv_3 = Conv2DDNNLayer(cor_1_nl_2, name='cor_1_conv_3', num_filters=num_filters, filter_size=3, W=a_1_conv_3.W)
        cor_1_bn_3 = batch_norm_dnn(cor_1_conv_3)
        cor_1_nl_3 = prelu(cor_1_bn_3,  name = 'cor_1_nl_3')

        num_filters = 64
        cor_2_conv_1 = Conv2DDNNLayer(cor_1_nl_3, name='cor_2_conv_1', num_filters=num_filters, filter_size=3, W=a_2_conv_1.W)
        cor_2_bn_1 = batch_norm_dnn(cor_2_conv_1)
        cor_2_nl_1 = prelu(cor_2_bn_1,  name = 'cor_2_nl_1')
        cor_2_conv_2 = Conv2DDNNLayer(cor_2_nl_1, name='cor_2_conv_2', num_filters=num_filters, filter_size=3, W=a_2_conv_2.W)
        cor_2_bn_2 = batch_norm_dnn(cor_2_conv_2)
        cor_2_nl_2 = prelu(cor_2_bn_2,  name = 'cor_2_nl_2')
        cor_2_conv_3 = Conv2DDNNLayer(cor_2_nl_2, name='cor_2_conv_3', num_filters=num_filters, filter_size=3, W=a_2_conv_3.W)
        cor_2_bn_3 = batch_norm_dnn(cor_2_conv_3)
        cor_2_nl_3 = prelu(cor_2_bn_3,  name = 'cor_2_nl_3')
        
        num_filters = 128
        cor_3_conv_1 = Conv2DDNNLayer(cor_2_nl_3, name='cor_3_conv_1', num_filters=num_filters, filter_size=3, W=a_3_conv_1.W)
        cor_3_bn_1 = batch_norm_dnn(cor_3_conv_1)
        cor_3_nl_1 = prelu(cor_3_bn_1,  name = 'cor_3_nl_1')
        cor_3_conv_2 = Conv2DDNNLayer(cor_3_nl_1, name='cor_3_conv_2', num_filters=num_filters, filter_size=3, W=a_3_conv_2.W)
        cor_3_bn_2 = batch_norm_dnn(cor_3_conv_2)
        cor_3_nl_2 = prelu(cor_3_bn_2,  name = 'cor_3_nl_2')
        cor_3_conv_3 = Conv2DDNNLayer(cor_3_nl_2, name='cor_3_conv_3', num_filters=num_filters, filter_size=3, W=a_3_conv_3.W)
        cor_3_bn_3 = batch_norm_dnn(cor_3_conv_3)
        cor_3_nl_3 = prelu(cor_3_bn_3,  name = 'cor_3_nl_3')

        cor_4_fc = DenseLayer(cor_3_nl_3, name='cor_d1', num_units = fc_conv)
        cor_out = prelu(cor_4_fc, name = 'cor_fc_d1')



        # --------------------------------------------------
        # channel_1: coronal
        # --------------------------------------------------
        # input: 32

        sag_ch_input = InputLayer(name='in3', shape=(None, num_channels, ps, ps))

        num_filters = 32
        sag_1_conv_1 = Conv2DDNNLayer(sag_ch_input, name='sag_1_conv_1', num_filters=num_filters, filter_size=3, W=a_1_conv_1.W)
        sag_1_bn_1 = batch_norm_dnn(sag_1_conv_1)
        sag_1_nl_1 = prelu(sag_1_bn_1,  name = 'sag_1_nl_1')
        sag_1_conv_2 = Conv2DDNNLayer(sag_1_nl_1, name='sag_1_conv_2', num_filters=num_filters, filter_size=3, W=a_1_conv_2.W)
        sag_1_bn_2 = batch_norm_dnn(sag_1_conv_2)
        sag_1_nl_2 = prelu(sag_1_bn_2,  name = 'sag_1_nl_2')
        sag_1_conv_3 = Conv2DDNNLayer(sag_1_nl_2, name='sag_1_conv_3', num_filters=num_filters, filter_size=3, W=a_1_conv_3.W)
        sag_1_bn_3 = batch_norm_dnn(sag_1_conv_3)
        sag_1_nl_3 = prelu(sag_1_bn_3,  name = 'sag_1_nl_3')

        num_filters = 64
        sag_2_conv_1 = Conv2DDNNLayer(sag_1_nl_3, name='sag_2_conv_1', num_filters=num_filters, filter_size=3, W=a_2_conv_1.W)
        sag_2_bn_1 = batch_norm_dnn(sag_2_conv_1)
        sag_2_nl_1 = prelu(sag_2_bn_1,  name = 'sag_2_nl_1')
        sag_2_conv_2 = Conv2DDNNLayer(sag_2_nl_1, name='sag_2_conv_2', num_filters=num_filters, filter_size=3, W=a_2_conv_2.W)
        sag_2_bn_2 = batch_norm_dnn(sag_2_conv_2)
        sag_2_nl_2 = prelu(sag_2_bn_2,  name = 'sag_2_nl_2')
        sag_2_conv_3 = Conv2DDNNLayer(sag_2_nl_2, name='sag_2_conv_3', num_filters=num_filters, filter_size=3, W=a_2_conv_3.W)
        sag_2_bn_3 = batch_norm_dnn(sag_2_conv_3)
        sag_2_nl_3 = prelu(sag_2_bn_3,  name = 'sag_2_nl_3')
        
        num_filters = 128
        sag_3_conv_1 = Conv2DDNNLayer(sag_2_nl_3, name='sag_3_conv_1', num_filters=num_filters, filter_size=3, W=a_3_conv_1.W)
        sag_3_bn_1 = batch_norm_dnn(sag_3_conv_1)
        sag_3_nl_1 = prelu(sag_3_bn_1,  name = 'sag_3_nl_1')
        sag_3_conv_2 = Conv2DDNNLayer(sag_3_nl_1, name='sag_3_conv_2', num_filters=num_filters, filter_size=3, W=a_3_conv_2.W)
        sag_3_bn_2 = batch_norm_dnn(sag_3_conv_2)
        sag_3_nl_2 = prelu(sag_3_bn_2,  name = 'sag_3_nl_2')
        sag_3_conv_3 = Conv2DDNNLayer(sag_3_nl_2, name='sag_3_conv_3', num_filters=num_filters, filter_size=3, W=a_3_conv_3.W)
        sag_3_bn_3 = batch_norm_dnn(sag_3_conv_3)
        sag_3_nl_3 = prelu(sag_3_bn_3,  name = 'sag_3_nl_3')

        sag_4_fc = DenseLayer(sag_3_nl_3, name='sag_d1', num_units = fc_conv)
        sag_out = prelu(sag_4_fc, name = 'sag_fc_d1')


        # FC 
        layer = ConcatLayer(name = 'elem_channels', incomings = [a_out, cor_out, sag_out])

        layer = DropoutLayer(layer, name = 'f1_drop', p = dropout_fc)        
        layer = DenseLayer(layer, name='FC1', num_units =768)
        layer = prelu(layer, name = 'prelu_f1')
        layer = DropoutLayer(layer, name = 'f2_drop', p = dropout_fc)        

        
        # concatenate channels
        atlas_layer = DropoutLayer(InputLayer(name='in4', shape=(None, 15)), name = 'Dropout_atlas', p = .1)
        atlas_layer = InputLayer(name='in4', shape=(None, 15))
        layer = ConcatLayer(name = 'elem_channels2', incomings = [layer, atlas_layer])

        # fully connected layer
        layer = DenseLayer(layer, name='fc_2', num_units = 384)
        layer = prelu(layer, name = 'prelu_f2')
        
        # softmax
        net_layer = DenseLayer(layer, name='out_layer', num_units = 15, nonlinearity=softmax)

        net =  NeuralNet(
            layers= net_layer,
            objective_loss_function=objectives.categorical_crossentropy,
            update = updates.adam,
            update_learning_rate = T.shared(float32(0.0001)),
            on_epoch_finished=[
                        save_weights,
                        save_training_history,
                        early_stopping,
                        AdjustVariable('update_learning_rate', start=0.0001, stop=0.000001),                
            ],
            verbose= t_verbose,
            max_epochs= num_epochs,
            train_split=TrainSplit(eval_size= train_split_perc),
        )


    return net
