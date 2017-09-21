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

This subcortical brain tissue segmentation method only relies on T1-w images. For training,  manual sub-cortical labels have to be provided along with T1-w images. When training, the method expects the data to be stored as:

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

All method parameters and options are easily tuned from the `configuration.cfg` file. 

## Dataset options: 

Dataset options are selected as follows, including `train_folder`, `inference_folder` and image names:  

```python
[database]
train_folder = /path/to/training_dataset/ 
inference_folder = /path/to/images_to_infer_seg/
t1_name = t1_name #ex. t1.nii, t1.nii.gz
roi_name = gt_15_classes.nii.gz
save_tmp = True
```

Model options are selected as follows. Unless the model `name` and the training mode `mode`, the rest of parameters work well with default parameters.  

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


# Citing this work:

Please cite this work as:

```
put a publication 
```

 
