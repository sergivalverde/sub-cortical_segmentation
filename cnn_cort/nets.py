import os, argparse, cPickle
from shutil import copyfile
import numpy as np
import cPickle
from nibabel import load as load_nii
import nibabel as nib
from scipy import ndimage
from nolearn.lasagne import NeuralNet, BatchIterator, TrainSplit
from nolearn_utils.hooks import SaveTrainingHistory, PlotTrainingHistory, EarlyStopping
from lasagne import objectives, updates
from nolearn.lasagne.handlers import SaveWeights
import lasagne
import theano as T
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer, FeaturePoolLayer, LocalResponseNormalization2DLayer, BatchNormLayer, prelu, ConcatLayer, ElemwiseSumLayer, ExpressionLayer, PadLayer, ScaleLayer
from lasagne.layers import Conv3DLayer, MaxPool3DLayer, Conv2DLayer, MaxPool2DLayer, Pool3DLayer, batch_norm
from lasagne.nonlinearities import softmax, rectify
nib.Nifti1Header.quaternion_threshold = -np.finfo(np.float32).eps * 10
from datetime import datetime


def float32(k):
        return np.cast['float32'](k)


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


def build_model(weights_path, options):
    """
    Build the CNN model. Create the Neural Net object and return it back. 
    Inputs: 
    - subject name: used to save the net weights accordingly.
    - options: several hyper-parameters used to configure the net.
    
    Output:
    - net: a NeuralNet object 
    """

    net_model_name = options['experiment']

    try:
        os.mkdir(os.path.join(weights_path, net_model_name))
    except:
        pass


    net_weights = os.path.join(weights_path, net_model_name, net_model_name + '.pkl')
    net_history = os.path.join(weights_path, net_model_name, net_model_name + '_history.pkl')
    
    # select hyper-parameters
    t_verbose = options['net_verbose']  
    train_split_perc = options['train_split']
    num_epochs = options['max_epochs']
    max_epochs_patience = options['patience']
    early_stopping = EarlyStopping(patience=max_epochs_patience)
    save_weights = SaveWeights(net_weights, only_best=True, pickle=False)
    save_training_history = SaveTrainingHistory(net_history)

    # build the architecture 
    ps = options['patch_size'][0]
    num_channels = 1
    fc_conv = 180
    fc_fc = 180 
    dropout_conv = 0.5
    dropout_fc = 0.5
        
    # --------------------------------------------------
    # channel_1: axial
    # --------------------------------------------------
    
    axial_ch = InputLayer(name='in1', shape=(None, num_channels, ps, ps))
    axial_ch = prelu(batch_norm(Conv2DLayer(axial_ch, name='axial_ch_conv1', num_filters=20, filter_size=3)),  name = 'axial_ch_prelu1')
    axial_ch = prelu(batch_norm(Conv2DLayer(axial_ch, name='axial_ch_conv2', num_filters=20, filter_size=3)),  name = 'axial_ch_prelu2')
    axial_ch = MaxPool2DLayer(axial_ch, name='axial_max_pool_1', pool_size=2)
    axial_ch = prelu(batch_norm(Conv2DLayer(axial_ch, name='axial_ch_conv3', num_filters=40, filter_size=3)),  name = 'axial_ch_prelu3')
    axial_ch = prelu(batch_norm(Conv2DLayer(axial_ch, name='axial_ch_conv4', num_filters=40, filter_size=3)),  name = 'axial_ch_prelu4')
    axial_ch = MaxPool2DLayer(axial_ch, name='axial_max_pool_2', pool_size=2)
    axial_ch = prelu(batch_norm(Conv2DLayer(axial_ch, name='axial_ch_conv5', num_filters=60, filter_size=3)),  name = 'axial_ch_prelu5')
    axial_ch = DropoutLayer(axial_ch, name = 'axial_l1drop', p = dropout_conv)
    axial_ch = DenseLayer(axial_ch, name='axial_d1', num_units = fc_conv)
    axial_ch = prelu(axial_ch, name = 'axial_prelu_d1')

    # --------------------------------------------------
    # channel_1: coronal
    # --------------------------------------------------

    coronal_ch = InputLayer(name='in2', shape=(None, num_channels, ps, ps))
    coronal_ch = prelu(batch_norm(Conv2DLayer(coronal_ch, name='coronal_ch_conv1', num_filters=20, filter_size=3)),  name = 'coronal_ch_prelu1')
    coronal_ch = prelu(batch_norm(Conv2DLayer(coronal_ch, name='coronal_ch_conv2', num_filters=20, filter_size=3)),  name = 'coronal_ch_prelu2')
    coronal_ch = MaxPool2DLayer(coronal_ch, name='coronal_max_pool_1', pool_size=2)
    coronal_ch = prelu(batch_norm(Conv2DLayer(coronal_ch, name='coronal_ch_conv3', num_filters=40, filter_size=3)),  name = 'coronal_ch_prelu3')
    coronal_ch = prelu(batch_norm(Conv2DLayer(coronal_ch, name='coronal_ch_conv4', num_filters=40, filter_size=3)),  name = 'coronal_ch_prelu4')
    coronal_ch = MaxPool2DLayer(coronal_ch, name='coronal_max_pool_2', pool_size=2)
    coronal_ch = prelu(batch_norm(Conv2DLayer(coronal_ch, name='coronal_ch_conv5', num_filters=60, filter_size=3)),  name = 'coronal_ch_prelu5')
    coronal_ch = DropoutLayer(coronal_ch, name = 'coronal_l1drop', p = dropout_conv)
    coronal_ch = DenseLayer(coronal_ch, name='coronal_d1', num_units = fc_conv)
    coronal_ch = prelu(coronal_ch, name = 'coronal_prelu_d1')

    # --------------------------------------------------
    # channel_1: saggital
    # --------------------------------------------------

    saggital_ch = InputLayer(name='in3', shape=(None, num_channels, ps, ps))
    saggital_ch = prelu(batch_norm(Conv2DLayer(saggital_ch, name='saggital_ch_conv1', num_filters=20, filter_size=3)),  name = 'saggital_ch_prelu1')
    saggital_ch = prelu(batch_norm(Conv2DLayer(saggital_ch, name='saggital_ch_conv2', num_filters=20, filter_size=3)),  name = 'saggital_ch_prelu2')
    saggital_ch = MaxPool2DLayer(saggital_ch, name='saggital_max_pool_1', pool_size=2)
    saggital_ch = prelu(batch_norm(Conv2DLayer(saggital_ch, name='saggital_ch_conv3', num_filters=40, filter_size=3)),  name = 'saggital_ch_prelu3')
    saggital_ch = prelu(batch_norm(Conv2DLayer(saggital_ch, name='saggital_ch_conv4', num_filters=40, filter_size=3)),  name = 'saggital_ch_prelu4')
    saggital_ch = MaxPool2DLayer(saggital_ch, name='saggital_max_pool_2', pool_size=2)
    saggital_ch = prelu(batch_norm(Conv2DLayer(saggital_ch, name='saggital_ch_conv5', num_filters=60, filter_size=3)),  name = 'saggital_ch_prelu5')
    saggital_ch = DropoutLayer(saggital_ch, name = 'saggital_l1drop', p = dropout_conv)
    saggital_ch = DenseLayer(saggital_ch, name='saggital_d1', num_units = fc_conv)
    saggital_ch = prelu(saggital_ch, name = 'saggital_prelu_d1')
        
    # FC layer 540 
    layer = ConcatLayer(name = 'elem_channels', incomings = [axial_ch, coronal_ch, saggital_ch])
    layer = DropoutLayer(layer, name = 'f1_drop', p = dropout_fc)        
    layer = DenseLayer(layer, name='FC1', num_units =540)
    layer = prelu(layer, name = 'prelu_f1')
    
    # concatenate channels 540 + 15
    layer = DropoutLayer(layer, name = 'f2_drop', p = dropout_fc)        
    atlas_layer = DropoutLayer(InputLayer(name='in4', shape=(None, 15)), name = 'Dropout_atlas', p = .2)
    atlas_layer = InputLayer(name='in4', shape=(None, 15))
    layer = ConcatLayer(name = 'elem_channels2', incomings = [layer, atlas_layer])
    
    # FC layer 270
    layer = DenseLayer(layer, name='fc_2', num_units = 270)
    layer = prelu(layer, name = 'prelu_f2')
    
    # FC output 15 (softmax)
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

    if options['load_weights'] == 'True':
        try:
            print "    --> loading weights from ", net_weights
            net.load_params_from(net_weights)
        except:
            pass
        
    return net
