import os, re, gc
import numpy as np
from scipy import ndimage as nd
from nibabel import load as load_nii
import nibabel as nib
from math import floor
from operator import itemgetter
import h5py
import cPickle
import copy
from operator import add
nib.Nifti1Header.quaternion_threshold = -np.finfo(np.float32).eps * 10
import time 


def load_patch_vectors(name, label_name, dir_name, size, random_state=42, seeds = None, balance_neg = True, datatype=np.float32):
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
    - seeds: list of images with voxels classified as positive classes during the previous iteration

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
    images = [load_nii(name).get_data() for name in image_names] 
    images_norm = [(im.astype(dtype=datatype) - im[np.nonzero(im)].mean()) / im[np.nonzero(im)].std() for im in images]
    
    # load labels 
    label_names = [os.path.join(dir_name, subject, label_name) for subject in subjects]
    labels = [load_nii(name).get_data() for name in label_names]

    
    # positive classes (not background) classes between 1 and 14 
    p_vox_coord_pos = [get_mask_voxels(np.logical_and(mask>0, mask < 15)) for mask in labels]
    axial_x_pos_patches = [np.array(get_patches(image, centers, size, mode = 'axial')) for image, centers in zip(images_norm,  p_vox_coord_pos)]
    axial_y_pos_patches = [np.array(get_patches(image, centers, size, mode = 'axial')) for image, centers in zip(labels, p_vox_coord_pos)]
    cor_x_pos_patches = [np.array(get_patches(image, centers, size, mode = 'coronal')) for image, centers in zip(images_norm,  p_vox_coord_pos)]
    cor_y_pos_patches = [np.array(get_patches(image, centers, size, mode = 'coronal')) for image, centers in zip(labels, p_vox_coord_pos)]
    sag_x_pos_patches = [np.array(get_patches(image, centers, size, mode = 'saggital')) for image, centers in zip(images_norm,  p_vox_coord_pos)]
    sag_y_pos_patches = [np.array(get_patches(image, centers, size, mode = 'saggital')) for image, centers in zip(labels, p_vox_coord_pos)]
    
    # all negative are taken, as the GT includes only boundary voxels only
    if seeds is None:
        if balance_neg == True:
            n_vox_coord_pos = [get_mask_voxels(mask==15,  size=len(p)) for mask, p in zip(labels, p_vox_coord_pos)]
        else:
            n_vox_coord_pos = [get_mask_voxels(mask==15) for mask, p in zip(labels, p_vox_coord_pos)]        
    else:
        n_vox_coord_pos = [get_mask_voxels(np.logical_and(mask==15, seed ==1),  size=len(p)) for mask, seed, p in zip(labels, seeds, p_vox_coord_pos)]

    print "DEBUG: pos patches: ", len(np.concatenate(p_vox_coord_pos, axis=0))
    print "DEBUG: neg patches: ", len(np.concatenate(n_vox_coord_pos, axis=0))
        
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


def compute_neighbors(atlas_images, centers, radius = 1):
    """
    compute probabilities for each of the voxels
    TO DOC
    """
    
    t = time.time()
    size = [(radius*2)+1, (radius*2)+1]

    out = []

    all_patches = [np.stack([np.array(get_patches(patches[:,:,:,c], center, size, mode = 'axial')) for c in range(15)], axis = 1) for patches, center in zip(atlas_images, centers)]
    mean_patches = [np.mean(im.reshape([im.shape[0], im.shape[1], im.shape[2]*im.shape[3]]), axis = 2) for im in all_patches]
    print "DEBUGING NEIGHBORS:", len(mean_patches), mean_patches[0].shape, " --> estimated time: ", time.time() -t 

    return mean_patches 


def get_atlas_vectors(dir_name, current_scan, centers):

    """
    Generate training data vectors from probabilistic atlases. These vectors are concatenated with fully-connected layers.
    """

    subjects = [f for f in sorted(os.listdir(dir_name)) if os.path.isdir(os.path.join(dir_name, f))]
    atlas_names =  [os.path.join(dir_name, subject, 'mni_atlas', 'MNI_sub_prob_def.nii.gz') for subject in subjects]
    atlas_images =  [load_nii(atlas).get_data() for atlas in atlas_names]
    #atlas_images =  [load_nii(atlas).get_data() for atlas in atlas_names]
    #atlas_names =  [os.path.join(dir_name, subject, 'CONV_135_135', subject + '_level_0_proba.nii.gz') for subject in subjects]
    
    # ATLAS probabilities (centered voxel)
    # convert lesion centers
    lc = map(lambda l: np.asarray(l), centers)
    atlas_vectors = [a[c[:,0], c[:,1], c[:,2]] for a, c in zip(atlas_images, lc)]

    # correct for background. if no probability exists for any class, set as background
    for index in range(len(atlas_vectors)):
        if np.sum(atlas_vectors[index]) == 0:
            atlas_vectors[v][14] = 1

    #  NEIGHBORS (1st class)
    # compute first level neighbors probability for each class
    #neigh_probabilities = compute_neighbors(atlas_images, centers, radius = 1)
    #neigh_probabilities_2 = compute_neighbors(atlas_images, centers, radius = 2)

    # merge both
    #out_prob = [np.concatenate([p1, p2], axis = 1) for p1, p2 in zip(atlas_vectors, neigh_probabilities)]
    #out_prob = [np.concatenate([p1, p2, p3], axis = 1) for p1, p2, p3 in zip(atlas_vectors, neigh_probabilities, neigh_probabilities_2)] 
    #print "DEBUGING NEIGHBORS:", len(out_prob), out_prob[0].shape
    return atlas_vectors 


def load_patches(dir_name, mask_name, t1_name, size, seeds = None, balance_neg = True):
    """
    Load all patches for a given subject image passed as argument. This function makes no sense when using only
    one channel, but it's useful when using more than one, as load_patch_vectors is called for each of the channels and 
    the outputs are stacked afterwards. 

    input: 
    - dir_name = absolute path of the database images
    - label_name: label name 
    - t1_name: T1 image name 
    - size: patch size [p1, p2]
    - seeds: list of images used as a seed 

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
    x_axial, y_axial, x_cor, y_cor, x_sag, y_sag, centers, t1_names = load_patch_vectors(t1_name,
                                                                                         mask_name,
                                                                                         dir_name,
                                                                                         size,
                                                                                         random_state,
                                                                                         seeds,
                                                                                         balance_neg)

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
        
    #image_names = np.stack([name for name in [t1_names] if name is not None])

    return t1_names


    
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
        indices_list = indices_list[:size]

    return indices_list


def load_patch_batch(image_name, batch_size, patch_size, pos_samples = None, dir_name = None, current_scan = None, datatype=np.float32):
    """
    Load testing data in batches to reduce RAM memory. Return data in batches. If a mask is passed as input
    only the voxels of this mask are considered for testing. This can be useful to test the cascade. 

    Input:
    - image_names: image modalities absolute paths 
    - batch_size: output batch size
    - patch_size: patch size in [p1, p2] 
    - pos_samples: a binary input mask of the input image with voxels classified as positive classes

    Output:
    - yields consecutive batches of testing patches:
       - x_axial [batch_size, num_channels, p1, p2]
       - x_coronal [batch_size, num_channels, p1, p2]
       - x_saggital [batch_size, num_channels, p1, p2]
       - x_atas [batch_size, 15]  
       - voxel coordinate
    """

    
    image = load_nii(image_name).get_data()
    image_norm = (image - image[np.nonzero(image)].mean()) / image[np.nonzero(image)].std()

    # take into account if a mask with positive samples is passed.
    # If not, extract all voxels 
    if pos_samples is None:
        lesion_centers = get_mask_voxels(image.astype(np.bool))
    else:
        lesion_centers = get_mask_voxels(pos_samples.astype(np.bool))

    # load atlas

    #atlas_name = os.path.join(dir_name, current_scan, 'build_atlas15', 'prior_atlas_4D_train_15_classes_' + current_scan + '.nii.gz')
    atlas_name = os.path.join(dir_name, current_scan, 'mni_atlas', 'MNI_sub_prob_def.nii.gz')
    #atlas_name = os.path.join(dir_name, current_scan, 'CONV_135_135', current_scan + '_level_0_proba.nii.gz')
    atlas_image =  load_nii(atlas_name).get_data()

    for i in range(0, len(lesion_centers), batch_size):
        
        centers = lesion_centers[i:i+batch_size]
        axial_patches = np.stack([np.array(get_patches(image_norm, centers, patch_size, mode= 'axial')).astype(datatype)],axis=1)
        coronal_patches = np.stack([np.array(get_patches(image_norm, centers, patch_size, mode= 'coronal')).astype(datatype)], axis=1)
        saggital_patches  = np.stack([np.array(get_patches(image_norm, centers, patch_size, mode= 'saggital')).astype(datatype)], axis = 1)

        # ATLAS probabilities
        cl = map(lambda l: np.asarray(l), centers)
        atlas_vector = np.stack([atlas_image[c[0],c[1],c[2]] for c in cl]).astype(dtype=np.float32)

        # correct for background
        for index in range(atlas_vector.shape[0]):
            if np.sum(atlas_vector[index]) == 0:
                atlas_vector[index,14] = 1

                
        # NEIGHBORS (first level)
        #size = [3,3]
        #all_patches = np.stack([get_patches(atlas_image[:,:,:,c], centers, size, mode = 'axial') for c in range(15)], axis = 1)        
        #mean_patches = np.mean(all_patches.reshape([all_patches.shape[0], all_patches.shape[1], all_patches.shape[2]*all_patches.shape[3]]), axis = 2)

        # NEIGHBORS (second level)
        # size = [5,5]
        # all_patches = np.stack([get_patches(atlas_image[:,:,:,c], centers, size, mode = 'axial') for c in range(15)], axis = 1)        
        # mean_patches_2 = np.mean(all_patches.reshape([all_patches.shape[0], all_patches.shape[1], all_patches.shape[2]*all_patches.shape[3]]), axis = 2)


        # out_prob = np.stack([np.concatenate([p1, p2, p3], axis = 0) for p1, p2, p3  in zip(atlas_vector, mean_patches, mean_patches_2)], axis = 0)

               
        yield axial_patches, coronal_patches, saggital_patches, atlas_vector, centers

        


        
        
