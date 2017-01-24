# ++++++++++++++++++++++++++++++++++++++++++++++++++
# network configuration file
#
# ++++++++++++++++++++++++++++++++++++++++++++++++++

import os
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer, FeaturePoolLayer, LocalResponseNormalization2DLayer, BatchNormLayer, prelu, ConcatLayer, ElemwiseSumLayer
from lasagne.layers.dnn import Conv3DDNNLayer, MaxPool3DDNNLayer, Conv2DDNNLayer, MaxPool2DDNNLayer, Pool3DDNNLayer, batch_norm_dnn
from lasagne.nonlinearities import softmax, rectify
'''
ps = 15
num_channels = 1
# channel_1: axial
axial_ch = InputLayer(name='in1', shape=(None, num_channels, ps, ps))
axial_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(axial_ch, name='axial_ch_conv1', num_filters=20, filter_size=3)),  name = 'axial_ch_prelu1')
axial_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(axial_ch, name='axial_ch_conv2', num_filters=20, filter_size=3)),  name = 'axial_ch_prelu2')
axial_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(axial_ch, name='axial_ch_conv3', num_filters=40, filter_size=3)),  name = 'axial_ch_prelu3')
axial_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(axial_ch, name='axial_ch_conv4', num_filters=40, filter_size=3)),  name = 'axial_ch_prelu4')
axial_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(axial_ch, name='axial_ch_conv5', num_filters=60, filter_size=3)),  name = 'axial_ch_prelu5')
axial_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(axial_ch, name='axial_ch_conv6', num_filters=60, filter_size=3)),  name = 'axial_ch_prelu6')
#axial_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(axial_ch, name='axial_ch_conv7', num_filters=80, filter_size=3)),  name = 'axial_ch_prelu7')
#axial_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(axial_ch, name='axial_ch_conv8', num_filters=80, filter_size=3)),  name = 'axial_ch_prelu8')
#axial_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(axial_ch, name='axial_ch_conv9', num_filters=80, filter_size=3)),  name = 'axial_ch_prelu9')
#axial_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(axial_ch, name='axial_ch_conv10', num_filters=100, filter_size=3)),  name = 'axial_ch_prelu10')
#axial_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(axial_ch, name='axial_ch_conv11', num_filters=100, filter_size=3)),  name = 'axial_ch_prelu11')
#axial_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(axial_ch, name='axial_ch_conv12', num_filters=100, filter_size=3)),  name = 'axial_ch_prelu12')
axial_ch = DenseLayer(axial_ch, name='axial_d1', num_units = 120)
axial_ch = prelu(axial_ch, name = 'axial_prelu_d1')
axial_ch = DropoutLayer(axial_ch, name = 'axial_l1drop', p=0.5)

# channel 2: coronal
coronal_ch = InputLayer(name='in2', shape=(None, num_channels, ps, ps))
coronal_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(coronal_ch, name='coronal_ch_conv1', num_filters=20, filter_size=3)),  name = 'coronal_ch_prelu1')
coronal_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(coronal_ch, name='coronal_ch_conv2', num_filters=20, filter_size=3)),  name = 'coronal_ch_prelu2')
coronal_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(coronal_ch, name='coronal_ch_conv3', num_filters=40, filter_size=3)),  name = 'coronal_ch_prelu3')
coronal_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(coronal_ch, name='coronal_ch_conv4', num_filters=40, filter_size=3)),  name = 'coronal_ch_prelu4')
coronal_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(coronal_ch, name='coronal_ch_conv5', num_filters=60, filter_size=3)),  name = 'coronal_ch_prelu5')
coronal_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(coronal_ch, name='coronal_ch_conv6', num_filters=60, filter_size=3)),  name = 'coronal_ch_prelu6')
#coronal_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(coronal_ch, name='coronal_ch_conv7', num_filters=80, filter_size=3)),  name = 'coronal_ch_prelu7')
#coronal_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(coronal_ch, name='coronal_ch_conv8', num_filters=80, filter_size=3)),  name = 'coronal_ch_prelu8')
#coronal_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(coronal_ch, name='coronal_ch_conv9', num_filters=80, filter_size=3)),  name = 'coronal_ch_prelu9')
#coronal_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(coronal_ch, name='coronal_ch_conv10', num_filters=100, filter_size=3)),  name = 'coronal_ch_prelu10')
#coronal_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(coronal_ch, name='coronal_ch_conv11', num_filters=100, filter_size=3)),  name = 'coronal_ch_prelu11')
#coronal_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(coronal_ch, name='coronal_ch_conv12', num_filters=100, filter_size=3)),  name = 'coronal_ch_prelu12')
coronal_ch = DenseLayer(coronal_ch, name='coronal_d1', num_units = 120)
coronal_ch = prelu(coronal_ch, name = 'coronal_prelu_d1')
coronal_ch = DropoutLayer(coronal_ch, name = 'coronal_l1drop', p=0.3)
# channel 3: saggital 
saggital_ch = InputLayer(name='in3', shape=(None, num_channels, ps, ps))
saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv1', num_filters=20, filter_size=3)),  name = 'saggital_ch_prelu1')
saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv2', num_filters=20, filter_size=3)),  name = 'saggital_ch_prelu2')
saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv3', num_filters=40, filter_size=3)),  name = 'saggital_ch_prelu3')
saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv4', num_filters=40, filter_size=3)),  name = 'saggital_ch_prelu4')
saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv5', num_filters=60, filter_size=3)),  name = 'saggital_ch_prelu5')
saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv6', num_filters=60, filter_size=3)),  name = 'saggital_ch_prelu6')
#saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv7', num_filters=80, filter_size=3)),  name = 'saggital_ch_prelu7')
#saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv8', num_filters=80, filter_size=3)),  name = 'saggital_ch_prelu8')
#saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv9', num_filters=80, filter_size=3)),  name = 'saggital_ch_prelu9')
#saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv10', num_filters=100, filter_size=3)),  name = 'saggital_ch_prelu10')
#saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv11', num_filters=100, filter_size=3)),  name = 'saggital_ch_prelu11')
#saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv12', num_filters=100, filter_size=3)),  name = 'saggital_ch_prelu12')
saggital_ch = DenseLayer(saggital_ch, name='saggital_d1', num_units = 120)
saggital_ch = prelu(saggital_ch, name = 'saggital_prelu_d1')
saggital_ch = DropoutLayer(saggital_ch, name = 'saggital_l1drop', p=0.3)

# concatenate channels
layer = ConcatLayer(name = 'conc_channels', incomings = [axial_ch, coronal_ch, saggital_ch])

# fully connected layer 
layer = DenseLayer(layer, name='fc_2', num_units = 120)
layer = prelu(layer, name = 'prelu_f2')
layer = DropoutLayer(layer, name = 'f2_drop', p=0.3)

# fully connected layer
#layer = DenseLayer(layer, name='fc_3', num_units = 64)
#layer = prelu(layer, name = 'prelu_f3')
#layer = DropoutLayer(layer, name = 'f3_drop', p=0.3)
# softmax
net_layer = DenseLayer(layer, name='out_layer', num_units = 15, nonlinearity=softmax)
'''

"""
ps = 29
num_channels = 1
# channel_1: axial
axial_ch = InputLayer(name='in1', shape=(None, num_channels, ps, ps))
axial_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(axial_ch, name='axial_ch_conv1', num_filters=20, filter_size=3)),  name = 'axial_ch_prelu1')
axial_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(axial_ch, name='axial_ch_conv2', num_filters=20, filter_size=3)),  name = 'axial_ch_prelu2')
axial_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(axial_ch, name='axial_ch_conv3', num_filters=40, filter_size=3)),  name = 'axial_ch_prelu3')
axial_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(axial_ch, name='axial_ch_conv4', num_filters=40, filter_size=3)),  name = 'axial_ch_prelu4')
axial_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(axial_ch, name='axial_ch_conv5', num_filters=60, filter_size=3)),  name = 'axial_ch_prelu5')
axial_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(axial_ch, name='axial_ch_conv6', num_filters=60, filter_size=3)),  name = 'axial_ch_prelu6')
axial_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(axial_ch, name='axial_ch_conv7', num_filters=80, filter_size=3)),  name = 'axial_ch_prelu7')
axial_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(axial_ch, name='axial_ch_conv8', num_filters=80, filter_size=3)),  name = 'axial_ch_prelu8')
axial_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(axial_ch, name='axial_ch_conv9', num_filters=80, filter_size=3)),  name = 'axial_ch_prelu9')
axial_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(axial_ch, name='axial_ch_conv10', num_filters=100, filter_size=3)),  name = 'axial_ch_prelu10')
axial_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(axial_ch, name='axial_ch_conv11', num_filters=100, filter_size=3)),  name = 'axial_ch_prelu11')
axial_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(axial_ch, name='axial_ch_conv12', num_filters=100, filter_size=3)),  name = 'axial_ch_prelu12')
axial_ch = DenseLayer(axial_ch, name='axial_d1', num_units = 300)
axial_ch = prelu(axial_ch, name = 'axial_prelu_d1')
axial_ch = DropoutLayer(axial_ch, name = 'axial_l1drop', p=0.5)

# channel 2: coronal
coronal_ch = InputLayer(name='in2', shape=(None, num_channels, ps, ps))
coronal_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(coronal_ch, name='coronal_ch_conv1', num_filters=20, filter_size=3)),  name = 'coronal_ch_prelu1')
coronal_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(coronal_ch, name='coronal_ch_conv2', num_filters=20, filter_size=3)),  name = 'coronal_ch_prelu2')
coronal_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(coronal_ch, name='coronal_ch_conv3', num_filters=40, filter_size=3)),  name = 'coronal_ch_prelu3')
coronal_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(coronal_ch, name='coronal_ch_conv4', num_filters=40, filter_size=3)),  name = 'coronal_ch_prelu4')
coronal_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(coronal_ch, name='coronal_ch_conv5', num_filters=60, filter_size=3)),  name = 'coronal_ch_prelu5')
coronal_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(coronal_ch, name='coronal_ch_conv6', num_filters=60, filter_size=3)),  name = 'coronal_ch_prelu6')
coronal_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(coronal_ch, name='coronal_ch_conv7', num_filters=80, filter_size=3)),  name = 'coronal_ch_prelu7')
coronal_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(coronal_ch, name='coronal_ch_conv8', num_filters=80, filter_size=3)),  name = 'coronal_ch_prelu8')
coronal_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(coronal_ch, name='coronal_ch_conv9', num_filters=80, filter_size=3)),  name = 'coronal_ch_prelu9')
coronal_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(coronal_ch, name='coronal_ch_conv10', num_filters=100, filter_size=3)),  name = 'coronal_ch_prelu10')
coronal_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(coronal_ch, name='coronal_ch_conv11', num_filters=100, filter_size=3)),  name = 'coronal_ch_prelu11')
coronal_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(coronal_ch, name='coronal_ch_conv12', num_filters=100, filter_size=3)),  name = 'coronal_ch_prelu12')
coronal_ch = DenseLayer(coronal_ch, name='coronal_d1', num_units = 300)
coronal_ch = prelu(coronal_ch, name = 'coronal_prelu_d1')
coronal_ch = DropoutLayer(coronal_ch, name = 'coronal_l1drop', p=0.5)
# channel 3: saggital 
saggital_ch = InputLayer(name='in3', shape=(None, num_channels, ps, ps))
saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv1', num_filters=20, filter_size=3)),  name = 'saggital_ch_prelu1')
saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv2', num_filters=20, filter_size=3)),  name = 'saggital_ch_prelu2')
saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv3', num_filters=40, filter_size=3)),  name = 'saggital_ch_prelu3')
saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv4', num_filters=40, filter_size=3)),  name = 'saggital_ch_prelu4')
saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv5', num_filters=60, filter_size=3)),  name = 'saggital_ch_prelu5')
saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv6', num_filters=60, filter_size=3)),  name = 'saggital_ch_prelu6')
saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv7', num_filters=80, filter_size=3)),  name = 'saggital_ch_prelu7')
saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv8', num_filters=80, filter_size=3)),  name = 'saggital_ch_prelu8')
saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv9', num_filters=80, filter_size=3)),  name = 'saggital_ch_prelu9')
saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv10', num_filters=100, filter_size=3)),  name = 'saggital_ch_prelu10')
saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv11', num_filters=100, filter_size=3)),  name = 'saggital_ch_prelu11')
saggital_ch = prelu(batch_norm_dnn(Conv2DDNNLayer(saggital_ch, name='saggital_ch_conv12', num_filters=100, filter_size=3)),  name = 'saggital_ch_prelu12')
saggital_ch = DenseLayer(saggital_ch, name='saggital_d1', num_units = 300)
saggital_ch = prelu(saggital_ch, name = 'saggital_prelu_d1')
saggital_ch = DropoutLayer(saggital_ch, name = 'saggital_l1drop', p=0.5)

# concatenate channels
layer = ElemwiseSumLayer(name = 'elem_channels', incomings = [axial_ch, coronal_ch, saggital_ch])

# fully connected layer 
layer = DenseLayer(layer, name='fc_2', num_units = 150)
layer = prelu(layer, name = 'prelu_f2')
layer = DropoutLayer(layer, name = 'f2_drop', p=0.3)

# fully connected layer
#layer = DenseLayer(layer, name='fc_3', num_units = 64)
#layer = prelu(layer, name = 'prelu_f3')
#layer = DropoutLayer(layer, name = 'f3_drop', p=0.3)
# softmax
net_layer = DenseLayer(layer, name='out_layer', num_units = 15, nonlinearity=softmax)
"""

ps = 32
num_channels = 1

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
axial_ch = DenseLayer(axial_ch, name='axial_d1', num_units = 60)
axial_ch = prelu(axial_ch, name = 'axial_prelu_d1')
axial_ch = DropoutLayer(axial_ch, name = 'axial_l1drop', p=0.5)

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
coronal_ch = DenseLayer(coronal_ch, name='coronal_d1', num_units = 60)
coronal_ch = prelu(coronal_ch, name = 'coronal_prelu_d1')
coronal_ch = DropoutLayer(coronal_ch, name = 'coronal_l1drop', p=0.5)

# --------------------------------------------------
# channel_1: saggital
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
saggital_ch = DenseLayer(saggital_ch, name='saggital_d1', num_units = 60)
saggital_ch = prelu(saggital_ch, name = 'saggital_prelu_d1')
saggital_ch = DropoutLayer(saggital_ch, name = 'saggital_l1drop', p=0.5)

# concatenate channels
layer = ConcatLayer(name = 'elem_channels', incomings = [axial_ch, coronal_ch, saggital_ch])

# fully connected layer 
layer = DenseLayer(layer, name='fc_2', num_units = 60)
layer = prelu(layer, name = 'prelu_f2')
layer = DropoutLayer(layer, name = 'f2_drop', p=0.3)

# softmax
net_layer = DenseLayer(layer, name='out_layer', num_units = 15, nonlinearity=softmax)
