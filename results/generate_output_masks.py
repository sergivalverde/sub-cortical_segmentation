# --------------------------------------------------
# evaluate MICCAI 2012 results
#
# Sergi Valverde
# --------------------------------------------------

import numpy as np
import nibabel as nib
import os
from scipy.ndimage import binary_fill_holes
from scipy import ndimage
import time
from joblib import Parallel, delayed

def DSC(im1, im2):
    """
    dice coefficient 2nt/na + nb.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum


def preprocess_scan(input_segmentation, atlas):
    """
    preprocess a raw segentation. Take the region with maximum overlap
    with respect to the atlas 
    """

    filtered_mask = np.zeros_like(input_segmentation)
    for l in range(1,15):

        #print "processing label ", l
        th_label = input_segmentation == l
        labels, num_labels = ndimage.label(th_label)

        # filter candidates by size. Only storing the biggest one
        label_list = np.unique(labels)
        num_elements_by_lesion = ndimage.labeled_comprehension(np.logical_and(th_label, atlas), labels, label_list, np.sum, float, 0)
        argmax = np.argmax(num_elements_by_lesion)

        # assign voxels to output
        current_voxels = np.stack(np.where(labels == argmax), axis =1)

        filtered_mask[current_voxels[:,0], current_voxels[:,1], current_voxels[:,2]] = l

    return filtered_mask



def generate_masks(options, f):
    """
    function to store the different masks 
    """
    # load probabilistic output maps
    seg_level_0 = nib.load(os.path.join(options['folder'], f, options['model'], f+'_level_0_proba.nii.gz')).get_data()
    seg_level_1 = nib.load(os.path.join(options['folder'], f, options['model'], f+'_level_1_proba.nii.gz')).get_data()

    # atlas scan masks used to find the sub-cortical images
    atlas_scan = nib.load(os.path.join(options['folder'], f, 'mni_atlas', 'MNI_mask_subcortical.nii.gz'))
    atlas = atlas_scan.get_data().astype(np.dtype('uint8'))

    gt_scan = nib.load(os.path.join(options['folder'], f, 'gt_15_classes.nii.gz'))
    GT = gt_scan.get_data()
    GT[GT == 15] = 0
    
    # save the results inside a subfolder
    if not os.path.exists(os.path.join(options['folder'], f, options['model'], 'generated_masks')):
        os.mkdir(os.path.join(options['folder'], f, options['model'], 'generated_masks'))

    # for each image compute four different outputs:
    # - level 1 with maximum probability
    # - level 1 with defined threshold
    # - level 2 with maximum probability
    # - level 2 with defined threshold

    # process and save maximum probability masks

    joint_seg = seg_level_0 * seg_level_1
    seg_it1_maxprob = np.argmax(seg_level_0, axis = 3).astype(np.dtype('uint8'))
    seg_cascade_maxprob = np.argmax(joint_seg, axis = 3).astype(np.dtype('uint8'))
    
    SEG_it1_maxprob = preprocess_scan(seg_it1_maxprob, atlas).astype(np.dtype('uint8'))

    SEG_cascade_maxprob = preprocess_scan(seg_cascade_maxprob, atlas).astype(np.dtype('uint8'))

    atlas_scan.get_data()[:] = SEG_it1_maxprob
    atlas_scan.to_filename(os.path.join(options['folder'], f, options['model'] , 'generated_masks', 'SEG_it1_maxprob.nii.gz'))
    atlas_scan.get_data()[:] = SEG_cascade_maxprob
    atlas_scan.to_filename(os.path.join(options['folder'], f, options['model'] , 'generated_masks', 'SEG_cascade_maxprob.nii.gz'))
    
    dsc_it1 = DSC(GT>0, SEG_it1_maxprob>0)
    dsc_cascade = DSC(GT>0, SEG_cascade_maxprob>0)
    print "----------------------------------------------------------"
    print f, 'iteration 1 (maxprob)', dsc_it1
    print f, 'cascade     (maxprob)', dsc_cascade

    
    # process and compute thresholded masks     

    for t in np.arange(0.1, 0.95, 0.05):

        seg_it1 = (seg_level_0 > t).astype(np.dtype('uint8'))
        seg_cascade = (joint_seg > t).astype(np.dtype('uint8'))
        seg_it1 = np.argmax(seg_it1, axis = 3).astype(np.dtype('uint8'))
        seg_cascade = np.argmax(seg_cascade, axis = 3).astype(np.dtype('uint8'))
    
        SEG_it1 = preprocess_scan(seg_it1, atlas).astype(np.dtype('uint8'))
        SEG_cascade = preprocess_scan(seg_cascade, atlas).astype(np.dtype('uint8'))

        atlas_scan.get_data()[:] = SEG_it1
        atlas_scan.to_filename(os.path.join(options['folder'], f, options['model'] , 'generated_masks', 'SEG_it1_'+str(t)+'.nii.gz'))
        atlas_scan.get_data()[:] = SEG_cascade
        atlas_scan.to_filename(os.path.join(options['folder'], f, options['model'] , 'generated_masks', 'SEG_cascade_'+str(t)+'.nii.gz'))


        dsc_it1 = DSC(GT>0, SEG_it1>0)
        dsc_cascade = DSC(GT>0, SEG_cascade>0)
        print f, 'iteration 1 (t', t, ') ', dsc_it1
        print f, 'cascade     (t', t, ') ', dsc_cascade
    
        
options  = {}
options['folder'] = '/mnt/DATA/w/CNN_CORT/images/MICCAI2012/test_set'
options['model'] = 'miccai_32_cascade_atlas'       
options['stucts'] = ['Thalamus R', 'Thalamus L', 'Caudate R', 'Caudate L', 'Putament R', 'Putamen L', 'Pallidum R', 'Pallidum L', 'Hipo R', 'Hipo L', 'Amig R', 'Amig L', 'Acc R', 'Acc L']
options['show_struct_results'] = False

list_of_folders = os.listdir(options['folder'])
list_of_folders.sort(reverse=False)

# compute masks in parallel
num_cores = 9
results = Parallel(n_jobs=num_cores)(delayed(generate_masks)(options, f) for f in list_of_folders)





