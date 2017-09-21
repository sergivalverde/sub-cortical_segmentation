# Sub-cortical brain tissue segmentation of MRI

Subcortical brain tissue segmentation of MRI images using a combination of convolutional and a-priori spatial feautures. An electronic prepint of the method is available here:

```
Kushibar, K., Valverde, S., et al. Publication on arxiv
```

## Overview: 




## Install

The method works on top of [Lasagne](http://lasagne.readthedocs.io/en/latest/index.html) and [Theano](http://deeplearning.net/software/theano/). If the method is run using GPU, please be sure that the Theano ```cuda*``` backend has been installed [correctly](https://github.com/Theano/Theano/wiki/Converting-to-the-new-gpu-back-end%28gpuarray%29). In the case of CPU, be sure that the fast linear algebra [libraries](http://lasagne.readthedocs.io/en/latest/user/installation.html#numpy-scipy-blas) are also installed. 

Once these requirements are met, the rest of python libraries may be easily installed using ```pip```: 

```python
pip install -r requirements.txt 
```


# How to use it: 

This subcortical brain tissue segmentation method only relies on T1-w images. All method parameters and options are easily tuned from the `configuration.cfg` file. Dataset options are selected as follows, including `train_folder`, `inference_folder` and image names:  

```python
[database]
train_folder = /path/to/training_dataset/ 
inference_folder = /path/to/images_to_infer_seg/
t1_name = t1_name #ex. t1.nii, t1.nii.gz
roi_name = gt_15_classes.nii.gz
save_tmp = True
```


Unless the model `name` and the training mode `mode`, the rest of parameters work well in most of the situations with default parameters.  

```python
[model]
name = model_name (to reuse aftwerwards)
mode = cuda*  (select GPU number cuda0, cuda1 or cpu) 
patch_size = 32 
batch_size = 256 
patience = 20 (i.e maximum number of iterations of early-stopping)
net_verbose = 1 (show messages (1) or not (0))
max_epochs = 100 (number of epochs to train)
train_split = 0.25 (number of training samples used for validation)
test_batch_size = 100000 (number of samples for each testing batch)
load_weights = True (permit the model to load the content of trained model)
out_probabilities = False (Compute probability maps for each output class)
speedup_segmentation = True (Reduce the segmentation only to the subcortical space)
post_process = True (Postprocess the segmentation excluding sporious regions)
debug = True (Show messsages that can be useful for degugging the model)
```


## Training: 

For training,  manual sub-cortical labels have to be provided along with T1-w images. When training, the method expects the data to be stored as:

```
[training_folder]
	     [image_1_folder]
			    t1_image
				manual_annotation (roi)
	     [image_2_folder]
			    t1_image
				manual_annotation (roi)
	     
		 ....
		 
		 
	     [image_n_folder]
			    t1_image
				manual_annotation (roi)

```

### Example: 

Load the configuration file from the `configuration.cfg` file:

```python

import os, sys, ConfigParser
import nibabel as nib
from cnn_cort.load_options import *


user_config = ConfigParser.RawConfigParser()
user_config.read(os.path.join(os.getcwd(), 'configuration.cfg'))
options = load_options(user_config)

```
Then, access data according to the selected `train_folder` building a set training patches to be used for training: 

```python
from cnn_cort.base import load_data, generate_training_set,


# get data patches from all orthogonal views 
x_axial, x_cor, x_sag, y, x_atlas, names = load_data(options)

# build the training dataset
x_train_axial, x_train_cor, x_train_sag, x_train_atlas, y_train = generate_training_set(x_axial,
                                                                                        x_cor,
                                                                                        x_sag,
                                                                                        x_atlas,
                                                                                        y,
                                                                                        options)
```

Once data patches are extracted, the network model can be compiled. The best network weights are stored in `weights_path`. These weights can be then accessed for future inference. For a more concise description of the network structure and details please visit the related publication.

```python

from cnn_cort.nets import build_model
weights_path = os.path.join(CURRENT_PATH, 'nets')
net = build_model(weights_path, options)

# train the net
net.fit({'in1': x_train_axial,
         'in2': x_train_cor,
         'in3': x_train_sag,
         'in4': x_train_atlas}, y_train)

```

## Testing 

Once a trained model exist, this can be used to infer sub-cortical classes on other images of the same image domain. The next example shows a simple script for batch-processing. We assume here that testing images follow also the same folder structure seen for training: 

```
[testing_folder]
	     [image_1_folder]
			    t1_image

	     [image_2_folder]
			    t1_image	     
		 ....
		 
	     [image_n_folder]
			    t1_image
```


### Example: 

First, we load the paths of all the images to infer: 

```python 
from cnn_cort.base import load_test_names, test_scan 

# get the testing image paths
t1_test_paths, folder_names  = load_test_names(options)
```

Then, the trained network model set in the configuration as (`name`) is loaded with the best trained weights. We assume that the `load_weights` option is set to `True` in the `configuration.cfg` file.


```
from cnn_cort.nets import build_model

weights_path = os.path.join(CURRENT_PATH, 'nets')
net = build_model(weights_path, options)
```

Finally, for each image to test, we call the `test_scan` function that will save the final segmentation on the same folder than the input image: 


```
# iterate through all test scans
for t1, current_scan in zip(t1_test_paths, folder_names):
    t = test_scan(net, t1, options)
    print "    -->  tested subject :", current_scan, "(elapsed time:", t, "min.)"
```


# Citing this work:

Please cite this work as:

```
put a publication 
```

 
