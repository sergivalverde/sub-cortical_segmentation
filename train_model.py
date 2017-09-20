import os, sys.ConfigParser
impot nibabel as nib
from cnn_cort.load_options import *


CURRENT_PATH = os.getcwd()


# --------------------------------------------------
# load options from config file
# --------------------------------------------------

user_config = ConfigParser.RawConfigParser()

user_config.read(os.path.join(CURRENT_PATH, 'configuration.cfg'))
options = load_options(default_config, user_config)

if options['mode'].find('cuda') == -1:
    os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=cpu,floatX=float32,optimizer=fast_compile'
else:
    os.environ['THEANO_FLAGS']='mode=FAST_RUN,device='+options['mode'] +',floatX=float32,optimizer=fast_compile'

# --------------------------------------------------
# load data 
# --------------------------------------------------

from cnn_cort.base import load_data, generate_training_set, load_test_names, load_patch_batch
from cnn_cort.nets import build_model
                 
# get data patches from all orthogonal views 
x_axial, x_cor, x_sag, y, x_atlas, names = load_data(options)

# build the training dataset
x_train_axial, x_train_cor, x_train_sag, x_train_atlas, y_train = generate_training_set(x_axial,
                                                                                        x_cor,
                                                                                        x_sag,
                                                                                        y,
                                                                                        x_atlas,
                                                                                        options)

# --------------------------------------------------
# build the net model
# --------------------------------------------------
weights_path = os.path.join(current_PATH, 'nets')
net = build_model(weights_path, options)


# --------------------------------------------------
# train the net
# --------------------------------------------------
net.fit({'in1': x_train_axial,
         'in2': x_train_cor,
         'in3': x_train_sag,
         'in4': x_train_atlas}, y)



# --------------------------------------------------
# test the model (for each scan)
# --------------------------------------------------

# get the testing image paths
t1_test_paths, folder_names  = load_test_names(options)

# reload the network weights and build it 
weights_path = os.path.join(current_PATH, 'nets')
net = build_model(weights_path, options)

# iterate through all test scans
for t1, current_scan in zip(t1_test_paths, folder_names):
    
    print "    -->  testing on subject :", current_scan
    test_scan(net, t1, options)
    
    

