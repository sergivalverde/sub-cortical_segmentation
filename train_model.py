# ------------------------------------------------------------
# Training script example
#
# - Useful to train and test a single network (ex. MICCAI2012)
#
# Sergi Valverde 2017
# svalverde@eia.udg.edu
# ------------------------------------------------------------

import os, sys, ConfigParser
import nibabel as nib
from cnn_cort.load_options import *


CURRENT_PATH = os.getcwd()


# --------------------------------------------------
# 1. load options from config file. Options are set
#    the configuration.cfg file 
# --------------------------------------------------

user_config = ConfigParser.RawConfigParser()

user_config.read(os.path.join(CURRENT_PATH, 'configuration.cfg'))
options = load_options(user_config)

# --------------------------------------------------
# load data 
# --------------------------------------------------

from cnn_cort.base import load_data, generate_training_set, load_test_names, test_scan
from cnn_cort.nets import build_model

'''
# get data patches from all orthogonal views 
x_axial, x_cor, x_sag, y, x_atlas, names = load_data(options)

# build the training dataset
x_train_axial, x_train_cor, x_train_sag, x_train_atlas, y_train = generate_training_set(x_axial,
                                                                                        x_cor,
                                                                                        x_sag,
                                                                                        x_atlas,
                                                                                        y,
                                                                                        options)

# --------------------------------------------------
# build the net model
# --------------------------------------------------
weights_path = os.path.join(CURRENT_PATH, 'nets')
net = build_model(weights_path, options)


# --------------------------------------------------
# train the net
# --------------------------------------------------
net.fit({'in1': x_train_axial,
         'in2': x_train_cor,
         'in3': x_train_sag,
         'in4': x_train_atlas}, y_train)

'''
# --------------------------------------------------
# test the model (for each scan)
# --------------------------------------------------

# get the testing image paths
t1_test_paths, folder_names  = load_test_names(options)

# reload the network weights and build it 
weights_path = os.path.join(CURRENT_PATH, 'nets')
options['net_verbose'] = 0
net = build_model(weights_path, options)

# iterate through all test scans
for t1, current_scan in zip(t1_test_paths, folder_names):
    t = test_scan(net, t1, options)
    print "    -->  tested subject :", current_scan, "(elapsed time:", t, "min.)"


    
    

