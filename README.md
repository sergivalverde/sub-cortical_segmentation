# Sub-cortical brain tissue segmentation of MRI

Subcortical brain tissue segmentation of MRI images using a combination of convolutional and a-priori spatial feautures. An electronic prepint of the method is available here: [https://arxiv.org/abs/1709.09075](https://arxiv.org/abs/1709.09075)


## Overview: 

This repository implements a voxelwise convolutional neural network based approach for accurate segmentation of the sub-cortical brain structures. In order to increase the sensitivity of the method on tissue boundaries and voxels with poor contrast or lack of contrast, we fuse a-priori class probabilities with convolutional features in the first fully convolutional layers of the net.

![pipeline](/imgs/pipeline.png)	


Contrary to other available works, the proposed network is trained using a restricted sample selection to force the network to learn only the most challenging voxels from structure boundaries. For more information about the method of the evaluation performed, please see the related publication. 


# Install:

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
weights_path = os.path.join(os.getcwd(), 'nets')
net = build_model(weights_path, options)

# train the net
net.fit({'in1': x_train_axial,
         'in2': x_train_cor,
         'in3': x_train_sag,
         'in4': x_train_atlas}, y_train)

```

## Testing:

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

Then, the trained network model set in the configuration as (`name`) is loaded with the best trained weights. We assume that the `load_weights` option is set to `True` in the `configuration.cfg` file:

```python
from cnn_cort.nets import build_model

weights_path = os.path.join(os.getcwd(), 'nets')
net = build_model(weights_path, options)
```

Finally, for each image to test, we call the `test_scan` function that will save the final segmentation on the same folder than the input image: 


```python
# iterate through all test scans
for t1, current_scan in zip(t1_test_paths, folder_names):
    t = test_scan(net, t1, options)
    print "    -->  tested subject :", current_scan, "(elapsed time:", t, "min.)"
```

The next figure depicts an output example, where the first image shows the output of the proposed CNN method and the second figure the manual annotation of subcortical structures also with the selected boundary voxels used as negatives (red): 

![pipeline](/imgs/example.png)	


# Limitations:

+ The current method would only work on GNU/Linux 64bit systems. We deal with image registration between spatial atlases and T1-w inside the training routines. So far, image registration is done using included binaries for [NiftiReg](http://niftyreg.sourceforge.net/) which have been compiled for GNU/Linux 64 bits. _We are working to move the method to Docker containers._


# Citing this work:

Please cite this work as:

```
@article{Kushibar2017,
abstract = {Sub-cortical brain structure segmentation in Magnetic Resonance Images (MRI) has attracted the interest of the research community for a long time because morphological changes in these structures are related to different neurodegenerative disorders. However, manual segmentation of these structures can be tedious and prone to variability, highlighting the need for robust automated segmentation methods. In this paper, we present a novel convolutional neural network based approach for accurate segmentation of the sub-cortical brain structures that combines both convolutional and prior spatial features for improving the segmentation accuracy. In order to increase the accuracy of the automated segmentation, we propose to train the network using a restricted sample selection to force the network to learn the most difficult parts of the structures. We evaluate the accuracy of the proposed method on the public MICCAI 2012 challenge and IBSR 18 datasets, comparing it with different available state-of-the-art methods and other recently proposed deep learning approaches. On the MICCAI 2012 dataset, our method shows an excellent performance comparable to the best challenge participant strategy, while performing significantly better than state-of-the-art techniques such as FreeSurfer and FIRST. On the IBSR 18 dataset, our method also exhibits a significant increase in the performance with respect to not only FreeSurfer and FIRST, but also comparable or better results than other recent deep learning approaches. Moreover, our experiments show that both the addition of the spatial priors and the restricted sampling strategy have a significant effect on the accuracy of the proposed method. In order to encourage the reproducibility and the use of the proposed method, a public version of our approach is available to download for the neuroimaging community.},
archivePrefix = {arXiv},
arxivId = {1709.09075},
author = {Kushibar, Kaisar and Valverde, Sergi and Gonzalez-Villa, Sandra and Bernal, Jose and Cabezas, Mariano and Oliver, Arnau and Llado, Xavier},
eprint = {1709.09075},
file = {:home/kaisar/Documents/MTReadings/1709.09075.pdf:pdf},
keywords = {brain,convolutional neural networks,mri,segmentation,sub-cortical structures},
title = {{Automated sub-cortical brain structure segmentation combining spatial and deep convolutional features}},
url = {http://arxiv.org/abs/1709.09075},
year = {2017}
} 
```

 
# License:

License for this software software: BSD 3-Clause License. A copy of this license is present in the root directory.
