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
from nolearn_utils.hooks import SaveTrainingHistory, PlotTrainingHistory #EarlyStopping
from lasagne import objectives, updates
#from nolearn.lasagne.handlers import SaveWeights
import lasagne
import theano as T
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer, FeaturePoolLayer, LocalResponseNormalization2DLayer, BatchNormLayer, prelu, ConcatLayer, ElemwiseSumLayer, ExpressionLayer, PadLayer, ScaleLayer
from lasagne.layers.dnn import Conv3DDNNLayer, MaxPool3DDNNLayer, Conv2DDNNLayer, MaxPool2DDNNLayer, Pool3DDNNLayer, batch_norm_dnn
from lasagne.nonlinearities import softmax, rectify
from skimage.transform import SimilarityTransform, warp, AffineTransform, rotate 
nib.Nifti1Header.quaternion_threshold = -np.finfo(np.float32).eps * 10
from datetime import datetime


class SaveWeights:
    def __init__(self, path, every_n_epochs=1, only_best=False,
                 pickle=False, verbose=0):
        self.path = path
        self.every_n_epochs = every_n_epochs
        self.only_best = only_best
        self.pickle = pickle
        self.verbose = verbose

    def __call__(self, nn, train_history):
        if self.only_best:
            this_acc = train_history[-1]['valid_accuracy']
            best_acc = min([h['valid_accuracy'] for h in train_history])
            if this_acc < best_acc:
                return

        if train_history[-1]['epoch'] % self.every_n_epochs != 0:
            return

        format_args = {
            'loss': train_history[-1]['valid_loss'],
            'timestamp': datetime.now().strftime('%Y-%m-%d-%H-%M-%S'),
            'epoch': '{:04d}'.format(train_history[-1]['epoch']),
            }
        path = self.path.format(**format_args)

        if self.verbose:
            print("Writing {}".format(path))

        if self.pickle:
            with open(path, 'wb') as f:
                pickle.dump(nn, f, -1)
        else:
            nn.save_params_to(path)

            
class EarlyStopping(object):
    """From https://github.com/dnouri/kfkd-tutorial"""
    def __init__(self, patience=50):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_valid_acc = train_history[-1]['valid_accuracy']
        current_train = train_history[-1]['train_loss']
        current_epoch = train_history[-1]['epoch']

        # Ignore if training loss is greater than valid loss
        if current_train > current_valid:
            return

        #if current_valid < self.best_valid:
        if current_valid_acc < self.best_valid:
            #self.best_valid = current_valid
            self.best_valid = current_valid_acc
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience <= current_epoch:
            print('Early stopping.')
            print('Best valid loss was {:.6f} at epoch {}.'.format(
                self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            raise StopIteration()


class weightAtlas(lasagne.layers.Layer):
    def __init__(self, incoming, num_units, W=lasagne.init.Normal(0.01), **kwargs):
        super(weightAtlas, self).__init__(incoming, **kwargs)
        num_inputs = self.input_shape[1]
        self.num_units = num_units
        self.W = self.add_param(W, (num_inputs,num_units), name='W')
        

    def get_output_for(self, input, **kwargs):
        return T.dot(input, self.W)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

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


def ceildiv(a, b):
    return -(-a // b)


def projection(l_inp):
    n_filters = l_inp.output_shape[1]*2
    l = ExpressionLayer(l_inp, lambda X: X[:, :, ::2, ::2], lambda s: (s[0], s[1], ceildiv(s[2], 2), ceildiv(s[3], 2)))
    l = PadLayer(l, [n_filters//4,0,0], batch_ndim=1)
    return l

def res_block(l_inp, name = None, increase_dim=False):
    """
    Residual Convolution block (He et al. 2015)
    # conv -> BN -> nonlin -> conv -> BN -> sum -> nonlin
    """
    
    # first figure filters/strides 
    n_filters, first_stride = filters_increase_dims(l_inp, increase_dim)

    # Convolution + batchnorm + prelu  
    l = prelu(batch_norm_dnn(Conv2DDNNLayer(l_inp, name = name+'_rl_conv1', num_filters=n_filters, filter_size=(3, 3),
                                            stride=first_stride, nonlinearity=None, pad='same',
                                            W=lasagne.init.HeNormal(gain='relu')), name = name+'_rl_batch_norm1'), name = name+'_rl_prelu1')

    # Convolution + batchnorm 
    l = batch_norm_dnn(Conv2DDNNLayer(l_inp, name = name+'_rl_conv2', num_filters=n_filters, filter_size=(3, 3),
                                            stride=first_stride, nonlinearity=None, pad='same',
                                            W=lasagne.init.HeNormal(gain='relu')), name = name+'_rl_batch_norm2')

    if increase_dim:
        # Use projection (A, B) as described in paper
        p = projection(l_inp)
    else:
        # Identity shortcut
        p = l_inp
    l = ElemwiseSumLayer([l, p], name= name+'_rl_sum')
    l = prelu(l, name = name+'_rl_batch_prelu2')
    
    return l



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
    
    if (options['experiment'] == 'CONV_135_135') or (options['experiment'] == 'CONV_135_135_DA') or (options['experiment'] == 'CONV_135_135_RE') or (options['experiment'] == 'CONV_135_135_DA_RE') or (options['experiment'] == 'CONV_135_135_ac'):
    
        fc_conv = 120
        fc_fc = (fc_conv)+15
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

    if (options['experiment'] == 'CONV_60_180_75_75'):
    
        fc_conv = 60
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
        layer = ConcatLayer(name = 'elem_channels_1', incomings = [axial_ch, coronal_ch, saggital_ch])

        # fully connected layer
        layer = DenseLayer(layer, name='fc_2', num_units = fc_fc)
        layer = prelu(layer, name = 'prelu_f2')
        layer = DropoutLayer(layer, name = 'f2_drop', p = dropout_fc)

        # fully connected layer
        atlas_layer = InputLayer(name='in4', shape=(None, 15))
        layer = ConcatLayer(name = 'elem_channels_2', incomings = [layer, atlas_layer])

        layer = DenseLayer(layer, name='fc_3', num_units = fc_fc+15)
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

        

    if (options['experiment'] == 'CONV25_60_195_195') or (options['experiment'] == 'CONV25_60_195_195_DA'):
    
        fc_conv = 60
        fc_fc = (fc_conv*3)+15
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

    if (options['experiment'] == 'CONV32_180_555_555') or (options['experiment'] == 'CONV32_180_555_555_DA'):
    
        fc_conv = 180 
        fc_fc = (fc_conv*3)+15
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


    if (options['experiment'] == 'CONVR32_80_195_80'):
    
        fc_conv = 80
        fc_fc = (fc_conv)+15
        dropout_conv = 0.4
        dropout_fc = 0.4
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
        axial_ch = DenseLayer(axial_ch, name='axial_dense_1', num_units = fc_conv)
        axial_ch = prelu(axial_ch, name = 'axial_prelu_dense_1')
        axial_ch = DropoutLayer(axial_ch, name = 'axial_dense_1_drop', p=dropout_conv)

        # --------------------------------------------------
        # channel_1: coronal
        # --------------------------------------------------

        # input: 32
        coronal_ch = InputLayer(name='in2', shape=(None, num_channels, ps, ps))
        coronal_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(coronal_ch, name='coronal_cnn_0', num_filters=init_num_filters, filter_size=3, pad = 'same')),  name = 'coronal_ch_prelu0')
        coronal_ch = res_block(coronal_ch, name = 'coronal_cnn_1', increase_dim=False)
        coronal_ch = res_block(coronal_ch, name = 'coronal_cnn_2', increase_dim=False)
        coronal_ch = res_block(coronal_ch, name = 'coronal_cnn_3', increase_dim=True)
        coronal_ch = res_block(coronal_ch, name = 'coronal_cnn_4', increase_dim=False)
        coronal_ch = res_block(coronal_ch, name = 'coronal_cnn_5', increase_dim=True)
        coronal_ch = res_block(coronal_ch, name = 'coronal_cnn_6', increase_dim=False)
        coronal_ch = DenseLayer(coronal_ch, name='coronal_dense_1', num_units = fc_conv)
        coronal_ch = prelu(coronal_ch, name = 'coronal_prelu_dense_1')
        coronal_ch = DropoutLayer(coronal_ch, name = 'coronal_dense_1_drop', p=dropout_conv)

        # --------------------------------------------------
        # channel_1: saggital
        # --------------------------------------------------

        # input: 32
        saggital_ch = InputLayer(name='in3', shape=(None, num_channels, ps, ps))
        saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_cnn_0', num_filters=init_num_filters, filter_size=3, pad = 'same')),  name = 'saggital_ch_prelu0')
        saggital_ch = res_block(saggital_ch, name = 'saggital_cnn_1', increase_dim=False)
        saggital_ch = res_block(saggital_ch, name = 'saggital_cnn_2', increase_dim=False)
        saggital_ch = res_block(saggital_ch, name = 'saggital_cnn_3', increase_dim=True)
        saggital_ch = res_block(saggital_ch, name = 'saggital_cnn_4', increase_dim=False)
        saggital_ch = res_block(saggital_ch, name = 'saggital_cnn_5', increase_dim=True)
        saggital_ch = res_block(saggital_ch, name = 'saggital_cnn_6', increase_dim=False)
        saggital_ch = DenseLayer(saggital_ch, name='saggital_dense_1', num_units = fc_conv)
        saggital_ch = prelu(saggital_ch, name = 'saggital_prelu_dense_1')
        saggital_ch = DropoutLayer(saggital_ch, name = 'saggital_dense_1_drop', p=dropout_conv)


        # concatenate channels
        atlas_layer = InputLayer(name='in4', shape=(None, 15))
        layer = ConcatLayer(name = 'elem_channels', incomings = [axial_ch, coronal_ch, saggital_ch, atlas_layer])

        # fully connected layer
        layer = DenseLayer(layer, name='fc_2', num_units = fc_fc)
        layer = prelu(layer, name = 'prelu_f2')
        layer = DropoutLayer(layer, name = 'f2_drop', p = dropout_fc)

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
        
    if (options['experiment'] == 'CONV_A15_135_135'):
        fc_conv = 120
        fc_fc = (fc_conv)+15
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


        # fully connected layer for atlas 
        a_layer = InputLayer(name='in4', shape=(None, 15))
        a_layer = DenseLayer(a_layer, name='fc_atlas', num_units = 15)
        a_layer = prelu(a_layer, name = 'prelu_atlas')
        a_layer = DropoutLayer(a_layer, name = 'atlas_drop', p = dropout_fc)

        # concatenate channels
        layer = ConcatLayer(name = 'elem_channels', incomings = [axial_ch, coronal_ch, saggital_ch, a_layer])

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

    if (options['experiment'] == 'CONV_135_135_sc') or (options['experiment'] == 'CONV_135_135_sc'):
    
        fc_conv = 120
        fc_fc = (fc_conv)+15
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
        #atlas_layer = weightAtlas(InputLayer(name='in4', shape=(None, 15)), num_units = 15,  name = 'atlas_weight', W=lasagne.init.Constant(1.0))
        atlas_layer = ScaleLayer(InputLayer(name='in4', shape=(None, 15)), name ='weightAtlas')
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



