import os, re, gc
import numpy as np
from scipy import ndimage as nd
from nibabel import load as load_nii
from math import floor
from operator import itemgetter
import h5py
import cPickle
import copy
from operator import add


def load_patch_vectors(name, label_name, dir_name, size, random_state=42, datatype=np.float32):
    """
    Generate all patch vectors for all subjects and one sequence (name). This is done for each image view (axial, coronal and axial)
    In subcortical brain tissue segmentation, I am extracting all positive class voxels (classes from 1 to 14) and the same number of
    negatives (background) voxels (class 15). 

    Inputs: 
    - name: T1 image name 
    - label_name: label name 
    - dir_name = absolute path of the database images
    - size: patch size [p1, p2]
    - random_state: random seed 

    Outputs:
    - x_axial: a list containing all the selected patches for all images for the axial view [image_num, num_samples, p1 , p2]
    - y_axial: a list containing all the labels for all image patches (axial view) [image_num, num_samples, p1 , p2]
    ...
    - y_saggital, a list containing all the labels for all image patches (saggital view) [image_num, num_samples, p1 , p2]
    - vox_positions: voxel coordinates for each of the patches [image_num, x, y, z]
    - image names 
    """    
    # Get the names of the images and load them and normalize images
    subjects = [f for f in sorted(os.listdir(dir_name)) if os.path.isdir(os.path.join(dir_name, f))]
    image_names = [os.path.join(dir_name, subject, name) for subject in subjects]
    images = [np.squeeze(load_nii(name).get_data()) for name in image_names] 
    images_norm = [(im.astype(dtype=datatype) - im[np.nonzero(im)].mean()) / im[np.nonzero(im)].std() for im in images]
    
    # load labels 
    label_names = [os.path.join(dir_name, subject, label_name) for subject in subjects]
    labels = [np.squeeze(load_nii(name).get_data()) for name in label_names]

    
    # positive classes (not background) classes between 1 and 14 
    p_vox_coord_pos = [get_mask_voxels(np.logical_and(mask>0, mask < 15)) for mask in labels]
    axial_x_pos_patches = [np.array(get_patches(image, centers, size, mode = 'axial')) for image, centers in zip(images_norm,  p_vox_coord_pos)]
    axial_y_pos_patches = [np.array(get_patches(image, centers, size, mode = 'axial')) for image, centers in zip(labels, p_vox_coord_pos)]
    cor_x_pos_patches = [np.array(get_patches(image, centers, size, mode = 'coronal')) for image, centers in zip(images_norm,  p_vox_coord_pos)]
    cor_y_pos_patches = [np.array(get_patches(image, centers, size, mode = 'coronal')) for image, centers in zip(labels, p_vox_coord_pos)]
    sag_x_pos_patches = [np.array(get_patches(image, centers, size, mode = 'saggital')) for image, centers in zip(images_norm,  p_vox_coord_pos)]
    sag_y_pos_patches = [np.array(get_patches(image, centers, size, mode = 'saggital')) for image, centers in zip(labels, p_vox_coord_pos)]

    # negatives class (background) class 15. sampling the same number of negatives of the rest of classes
    n_vox_coord_pos = [get_mask_voxels(mask==15,  size=x_pos_classes.shape[0]) for mask, x_pos_classes in zip(labels, axial_x_pos_patches)]
    axial_x_neg_patches = [np.array(get_patches(image, centers, size, mode = 'axial')) for image, centers in zip(images_norm, n_vox_coord_pos)]
    axial_y_neg_patches = [np.array(get_patches(image, centers, size, mode = 'axial')) for image, centers in zip(labels, n_vox_coord_pos)]
    cor_x_neg_patches = [np.array(get_patches(image, centers, size, mode = 'coronal')) for image, centers in zip(images_norm, n_vox_coord_pos)]
    cor_y_neg_patches = [np.array(get_patches(image, centers, size, mode = 'coronal')) for image, centers in zip(labels, n_vox_coord_pos)]
    sag_x_neg_patches = [np.array(get_patches(image, centers, size, mode = 'saggital')) for image, centers in zip(images_norm, n_vox_coord_pos)]
    sag_y_neg_patches = [np.array(get_patches(image, centers, size, mode = 'saggital')) for image, centers in zip(labels, n_vox_coord_pos)]
    
    
    x_axial = [np.concatenate([p1, p2]) for p1, p2 in zip(axial_x_pos_patches, axial_x_neg_patches)]
    y_axial = [np.concatenate([p1, p2]) for p1, p2 in zip(axial_y_pos_patches, axial_y_neg_patches)]
    x_cor = [np.concatenate([p1, p2]) for p1, p2 in zip(cor_x_pos_patches, cor_x_neg_patches)]
    y_cor = [np.concatenate([p1, p2]) for p1, p2 in zip(cor_y_pos_patches, cor_y_neg_patches)]
    x_sag = [np.concatenate([p1, p2]) for p1, p2 in zip(sag_x_pos_patches, sag_x_neg_patches)]
    y_sag = [np.concatenate([p1, p2]) for p1, p2 in zip(sag_y_pos_patches, sag_y_neg_patches)]
    vox_positions = [np.concatenate([p1, p2]) for p1, p2 in zip(p_vox_coord_pos, n_vox_coord_pos)]

    
    return x_axial, y_axial, x_cor, y_cor, x_sag, y_sag, vox_positions, image_names


"""
def get_atlas_vectors(dir_name, current_scan, centers):

    Generate training data vectors from probabilistic atlases. These vectors are concatenated with fully-connected layers.
   

    patients = [f for f in sorted(os.listdir(dir_name)) if os.path.isdir(os.path.join(dir_name, f))]
    
    atlas_names =  [os.path.join(dir_name, patient, 'build_atlas', 'prior_atlas_4D_train_' + current_scan + '.nii.gz') for patient in patients]
    atlas_images =  [load_nii(atlas).get_data() for atlas in atlas_names]

    # convert lesion centers
    lc = map(lambda l: np.asarray(l), centers)
    atlas_vectors = [a[c[:,0], c[:,1], c[:,2]] for a, c in zip(atlas_images, lc)]

    return atlas_vectors    
"""


def load_patches(dir_name, mask_name, t1_name, size):
    """
    Load all patches for a given subject image passed as argument. This function makes no sense when using only
    one channel, but it's useful when using more than one, as load_patch_vectors is called for each of the channels and 
    the outputs are stacked afterwards. 

    input: 
    - dir_name = absolute path of the database images
    - label_name: label name 
    - t1_name: T1 image name 
    - size: patch size [p1, p2]

    output:
    - x_axial: a list containing all selected patches (axial view) [num_samples, p1, p2]
    - y_axial a list containing all selected labels (axial view) [num_samples, p1, p2]
    - ...
    - y_axial a list containing all selected labels (saggital view) [num_samples, p1, p2]
    - centers: voxel coordinates for each patch.
    """
    # Setting up the lists for all images
    t1, t1_names = None, None
    y = None

    random_state = np.random.randint(1)


    print 'Loading ' + t1_name + ' images'
    x_axial, y_axial, x_cor, y_cor, x_sag, y_sag, centers, t1_names = load_patch_vectors(t1_name, mask_name, dir_name, size, random_state)
    
    #print 'Creating data vector'
    #data = [images for images in [t1] if images is not None]
    #flair, pd, t2, t1, gado = None, None, None, None, None
    #gc.collect()
    #print 'Stacking the numpy arrays'
    #x = [np.stack(images, axis=1) for images in zip(*data)]
    #data = None
    #gc.collect()
    #image_names = np.stack([name for name in [
    #    t1_names
    #] if name is not None])

    # reshape accordingly 
    return x_axial, y_axial, x_cor, y_cor, x_sag, y_sag, centers, t1_names 


def load_only_names(dir_name,mask_name,t1_name, use_t1, size):
    """
    Load image names given the options configuration file
    """
    patients = [f for f in sorted(os.listdir(dir_name)) if os.path.isdir(os.path.join(dir_name, f))]
    
    if use_t1:
        t1_names = [os.path.join(dir_name, patient, t1_name) for patient in patients]
        
    image_names = np.stack([name for name in [
        t1_names
    ] if name is not None])

    return image_names


    
def get_patches(image, centers, patch_size=(32, 32), mode = 'axial'):
    """
    Returns 2D patches of each of the image views (coronal, axial, sagittal), given a list of voxel coordinates (centers).

    Input: 
    - image: input image
    - centers: voxel coordinates
    - patch_size to generate the patches [p1, p2]
    - mode: 'axial', 'coronal' or 'saggital' to generate different view patches. 

    Output:
    - A list of patches for each voxel passed as input in "centers" --> [num_voxels, p1, p2]
    """
    # If the size has even numbers, the patch will be centered. If not, it will try to create an square almost centered.
    # By doing this we allow pooling when using encoders/unets.
    patches = []
    list_of_tuples = all([isinstance(center, tuple) for center in centers])
    #sizes_match = [len(center) == len(patch_size) for center in centers]
    
    # select between axial / coronal / saggital  patches
    if mode == 'axial':
        patch_size = (patch_size[0], patch_size[1], 1)
    if mode == 'coronal':
        patch_size = (patch_size[0], 1, patch_size[1])
    if mode == 'saggital':
        patch_size = (1, patch_size[0], patch_size[1])

    patch_half = tuple([idx/2 for idx in patch_size])
    new_centers = [map(add, center, patch_half) for center in centers]
    padding = tuple((idx, size-idx) for idx, size in zip(patch_half, patch_size))
    new_image = np.pad(image, padding, mode='constant', constant_values=0)
    
    slices = [[slice(c_idx-p_idx, c_idx+(s_idx-p_idx)) for (c_idx, p_idx, s_idx) in zip(center, patch_half, patch_size)] for center in new_centers]
    patches = [np.squeeze(new_image[idx]) for idx in slices]

    return patches


def get_mask_voxels(mask, size=None):
    """
    Return the voxel coordinates of non-zero voxels given a input image passed as argument.

    Input:
    - mask: Input image
    - size: if selected, return only a random list of length = size

    Output:
    - indices_list: a list of non-zero voxel positions expressed as a tuple [(x,y,z)]
    """
    import random
    indices = np.stack(np.nonzero(mask), axis=1)
    indices_list = [tuple(idx) for idx in indices]

    # if a output size is defined, shuffle and resize the list
    if size is not None:
        random.shuffle(indices_list)
        indices_list[:size]
    return indices_list



def load_patch_batch(image_names, batch_size, patch_size, datatype=np.float32):
    """
    Load testing data in batches to reduce RAM memory. Return data in batches. 

    Input:
    - image_names: image modalities absolute paths 
    - batch_size: output batch size
    - patch_size: patch size in [p1, p2] 

    Output:
    - yields consecutive batches of testing patches:
       - x_axial [batch_size, num_channels, p1, p2]
       - x_coronal [batch_size, num_channels, p1, p2]
       - x_saggital [batch_size, num_channels, p1, p2]
       - voxel coordinate
    """

    images = [np.squeeze(load_nii(name).get_data()) for name in image_names]
    images_norm = [(im - im[np.nonzero(im)].mean()) / im[np.nonzero(im)].std() for im in images]
    lesion_centers = get_mask_voxels(images[0].astype(np.bool))
    
    for i in range(0, len(lesion_centers), batch_size):
        centers = lesion_centers[i:i+batch_size]
        
        yield
        np.stack([np.array(get_patches(image, centers, patch_size, mode= 'axial')).astype(datatype) for image in images_norm], axis=1),
        np.stack([np.array(get_patches(image, centers, patch_size, mode= 'coronal')).astype(datatype) for image in images_norm], axis=1),
        np.stack([np.array(get_patches(image, centers, patch_size, mode= 'saggital')).astype(datatype) for image in images_norm], axis=1),
        centers

        


