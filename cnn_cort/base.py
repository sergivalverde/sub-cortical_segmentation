import os, cPickle, time
from shutil import copyfile
import numpy as np
import nibabel as nib
from nibabel import load as load_nii
import scipy.io as io
from scipy import ndimage, stats
from operator import add


def load_data(options):
    """
    Extact data from all images.  For all database image, patches from each image view (axial, coronal and saggital) are computed.
    This function is used to reduce the loading time in leave-one-out. So, data is only loaded one time and then training feature vectors
    for the classification of eac image (leave-one-out or others) are computed. 

    Input: 
    - options:
        - training folder (folder)
        - T1 name 
        - label name (mask)
        - patch size [p1, p2]

    - output:
      - x_axial: a list of X data (axial slice) indexed for each of the database subjects. Each element contains array of [features, p1, p2]
      - y_axial: a list of labels (axial slice) indexed for each of the database subjects. Each element contains array of [features, p1, p2]
      - ...
      - y_saggital: a list of labels (saggital slice) indexed for each of the database subjects. Each element contains array of [features, p1, p2]
      - centers: a list of voxel coordinates for each of the extracted patches for each of the database subjects. 
      - image names 
    """
    (x_axial, y_axial, x_cor, y_cor, x_sag, y_sag, x_atlas, names) = load_patches(dir_name=options['train_folder'],
                                                                                  t1_name=options['t1_name'],
                                                                                  mask_name=options['roi_name'],
                                                                                  size=tuple(options['patch_size']))

    return x_axial, x_cor, x_sag, y_axial, x_atlas, names



def load_test_names(options):
    """
    Load image names. Extract names from folders and return a list. 
    """
    dir_name = options['test_folder']
    t1_name = options['t1_name']
    subjects = [f for f in sorted(os.listdir(dir_name)) if os.path.isdir(os.path.join(dir_name, f))]
    t1_names = [os.path.join(dir_name, subject, t1_name) for subject in subjects]
        
    return t1_names, subjects


def generate_training_set(x_axial, x_coronal, x_saggital, x_atlas, y, options, randomize = True):
    """
    Generate training features X an Y for each image modality. 
   
    input: 
    - training folder: used to load the atlas
    - current_scan: used to load the atlas
    - i: index of the current scan (to remove from training)
    - x_axial: a list of X data (axial slice) indexed for each of the database subjects used for training. Each element contains array of [features, p1, p2]
    - ...
    - y_saggital: a list of labels (saggital slice) indexed for each of the database subjects used for training . Each element contains array of [features, p1, p2]
    - y_axial: a list of labels (axial slice) indexed for each of the database subjects used for training. Each element contains array of [features, p1, p2]
    - randomize: randomize training vectors

    output:
    - x_train_axial: axial training X   [num_samples, num_channels, p1, p2] 
    - x_train_cor: coronal training X   [num_samples, num_channels, p1, p2] 
    - x_train_sag: saggital training X  [num_samples, num_channels, p1, p2] 
    - y_train: labels. 
           * If voxelwise classification: [num_samples]
           * If fully-convolutional: [num_samples, 1, p1, p2] 
    - atlas: atlas vector containing the probability for each of the samples
    """

    # concatenate scans for training 
    x_train_axial = np.concatenate(x_axial, axis = 0).astype('float32')
    x_train_cor =   np.concatenate(x_coronal, axis = 0).astype('float32')
    x_train_sag =   np.concatenate(x_saggital, axis = 0).astype('float32')
    x_train_atlas = np.concatenate(x_atlas, axis = 0).astype('float32')
    y_train = np.concatenate(y, axis = 0).astype('uint8')
    
    # voxelwise:  convert labels to one-dimensional vector
    y_train = y_train[:, y_train.shape[1] / 2, y_train.shape[2] / 2]
    y_train = np.squeeze(y_train)

    # convert background to class 0 (generated as class = 15 in GT annotations)
    y_train[y_train==15] = 0

    
    if randomize:
        seed = np.random.randint(np.iinfo(np.int32).max)
        np.random.seed(seed)
        x_train_axial = np.random.permutation(x_train_axial)
        np.random.seed(seed)
        x_train_cor = np.random.permutation(x_train_cor)
        np.random.seed(seed)
        x_train_sag = np.random.permutation(x_train_sag)
        np.random.seed(seed)
        y_train = np.random.permutation(y_train)
        np.random.seed(seed)
        x_train_atlas = np.random.permutation(x_train_atlas)
    

    # The net expects training data with shape [samples, channels, p1, p2]
    # reshape arrays for single channel
    x_train_axial = np.expand_dims(x_train_axial, axis = 1)
    x_train_cor = np.expand_dims(x_train_cor, axis = 1)
    x_train_sag = np.expand_dims(x_train_sag, axis = 1)

    if options['debug'] == 'True':
        print "    --> X_TRAIN: ", x_train_axial.shape[0], x_train_axial.shape
        print "    --> Y_TRAIN POS: ", y_train[y_train > 0].shape[0]
        print "    --> Y_TRAIN NEG: ", y_train[y_train == 0].shape[0] 
    
    return x_train_axial, x_train_cor, x_train_sag, x_train_atlas, y_train


def load_patch_vectors(name, label_name, dir_name, size, random_state=42, balance_neg = True):
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
    images = [load_nii(name).get_data() for name in image_names] 
    images_norm = [(im.astype(np.float32) - im[np.nonzero(im)].mean()) / im[np.nonzero(im)].std() for im in images]
    
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
    if balance_neg == True:
        n_vox_coord_pos = [get_mask_voxels(mask==15,  size=len(p)) for mask, p in zip(labels, p_vox_coord_pos)]
    else:
        n_vox_coord_pos = [get_mask_voxels(mask==15) for mask, p in zip(labels, p_vox_coord_pos)]

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


def get_atlas_vectors(dir_name, centers, t1_names):

    """
    Generate training data vectors from probabilistic atlases. These vectors are concatenated with fully-connected layers.
    """

    subjects = [f for f in sorted(os.listdir(dir_name)) if os.path.isdir(os.path.join(dir_name, f))]
    atlas_names =  [os.path.join(dir_name, subject, 'tmp', 'MNI_sub_probabilities.nii.gz') for subject in subjects]

    
    atlas_images = []
    # load atlas, register if does not exist 
    for t1, atlas, subject in zip(t1_names, atlas_names, subjects):
        
        if os.path.exists(atlas) is False:
            print "         --> registering priors for scan", subject, 
            t = register_masks(t1)
            print "(elapsed time ", t/60.0, "min.)"
            
        atlas_images.append(load_nii(atlas).get_data())
                
    # ATLAS probabilities (centered voxel)
    # convert lesion centers
    lc = map(lambda l: np.asarray(l), centers)
    atlas_vectors = [a[c[:,0], c[:,1], c[:,2]] for a, c in zip(atlas_images, lc)]

    # correct for background. if no probability exists for any class, set as background
    for index in range(len(atlas_vectors)):
        if np.sum(atlas_vectors[index]) == 0:
            atlas_vectors[v][14] = 1

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

    random_state = np.random.randint(1)

    print '    --> Loading ' + t1_name + ' images'
    x_axial, y_axial, x_cor, y_cor, x_sag, y_sag, centers, t1_names = load_patch_vectors(t1_name,
                                                                                         mask_name,
                                                                                         dir_name,
                                                                                         size,
                                                                                         random_state)

    # load atlas vectors
    x_atlas = get_atlas_vectors(dir_name, centers, t1_names)
    
    return x_axial, y_axial, x_cor, y_cor, x_sag, y_sag, x_atlas, t1_names 


def load_only_names(dir_name,mask_name,t1_name, use_t1, size):
    """
    Load image names given the options configuration file
    """
    subjects = [f for f in sorted(os.listdir(dir_name)) if os.path.isdir(os.path.join(dir_name, f))]
    
    if use_t1:
        t1_names = [os.path.join(dir_name, subject, t1_name) for subject in subjects]
        
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



def load_patch_batch(scan_name,  options, datatype=np.float32):
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
    
    [dir_name, name] = os.path.split(scan_name)
    
    image = load_nii(scan_name).get_data()
    image_norm = (image - image[np.nonzero(image)].mean()) / image[np.nonzero(image)].std()


    if options['crop']:
        mask_atlas = nib.load(os.path.join(dir_name, 'tmp', 'MNI_subcortical_mask.nii.gz'))
        b_mask = ndimage.morphology.binary_dilation(mask_atlas.get_data(), iterations = 10)
        lesion_centers = get_mask_voxels(b_mask.astype(np.bool))
    else:
        lesion_centers = get_mask_voxels(image.astype(np.bool))

    if options['debug'] == 'True':
        print "    -->  num of samples to test:", len(lesion_centers)
        
    # load atlas, register if does not exist 
    atlas_name = os.path.join(dir_name,  'tmp', 'MNI_sub_probabilities.nii.gz')
    if os.path.exists(atlas_name) is False:
        print "         --> registering priors for scan", mame, 
        t = register_masks(scan_name)
        print "(elapsed time ", t/60.0, "min.)"

    atlas_image =  load_nii(atlas_name).get_data()
    batch_size = options['test_batch_size']
    for i in range(0, len(lesion_centers), batch_size):
        
        centers = lesion_centers[i:i+batch_size]
        axial_patches = np.stack([np.array(get_patches(image_norm, centers, options['patch_size'], mode= 'axial')).astype(datatype)],axis=1)
        coronal_patches = np.stack([np.array(get_patches(image_norm, centers, options['patch_size'], mode= 'coronal')).astype(datatype)], axis=1)
        saggital_patches  = np.stack([np.array(get_patches(image_norm, centers, options['patch_size'], mode= 'saggital')).astype(datatype)], axis = 1)

        # ATLAS probabilities
        cl = map(lambda l: np.asarray(l), centers)
        atlas_vector = np.stack([atlas_image[c[0],c[1],c[2]] for c in cl]).astype(dtype=np.float32)

        # correct for background
        for index in range(atlas_vector.shape[0]):
            if np.sum(atlas_vector[index]) == 0:
                atlas_vector[index,14] = 1

    
        yield axial_patches, coronal_patches, saggital_patches, atlas_vector, centers

        

def test_scan(net, test_scan, options):
    """
    Test scan function 
    
    to properly doc

    """

    s_time = time.time()
    [image_path, name] = os.path.split(test_scan)

    # create output images
    t1_nii = nib.load(test_scan)
    image = np.zeros_like(t1_nii.get_data())

    if options['out_probabilities'] == 'True':
        image_proba = np.zeros(t1_nii.shape + (15,))
            
    # test the image in batches to reduce the amount of required RAM
    # if options['crop'] is set, only a ROI around the subcortical space is infered 
    for batch_axial, batch_cor, batch_sag, atlas, centers in load_patch_batch(test_scan, options):
        if options['debug'] == 'True':

            # predict classes
            y_pred = net.predict({'in1': batch_axial,
                                'in2': batch_cor,
                                'in3': batch_sag,
                                'in4': atlas})

            [x, y, z] = np.stack(centers, axis=1)
            image[x, y, z] = y_pred

            # predict probabilities 
            if options['out_probabilities'] == 'True':
                y_pred_proba = net.predict_proba({'in1': batch_axial,
                                              'in2': batch_cor,
                                              'in3': batch_sag,
                                              'in4': atlas})
                for c in range(15):
                    image_proba[x, y, z, c] = y_pred_proba[:,c]

    # save segmentations


    if options['out_probabilities'] == 'True':        
        seg_out_prob = nib.Nifti1Image(image_proba, affine = t1_nii.affine)
        seg_out_prob.to_filename(os.path.join(image_path, 'out_subcortical_prob.nii.gz'))

    if options['post_process'] == 'True':
        filtered_nii = nib.Nifti1Image(post_process_segmentation(image_path, image),
                                       affine = t1_nii.affine)
        filtered_nii.to_filename(os.path.join(image_path, 'out_subcortical_seg_prec.nii.gz'))
    else:
        raw_nii = nib.Nifti1Image(image, affine = t1_nii.affine)                                        
        raw_nii.to_filename(os.path.join(image_path, 'out_subcortical_rawseg.nii.gz'))


    return (time.time() - s_time) / 60.0

def post_process_segmentation(image_folder, input_mask):
    """
    doc
    """
    filtered_mask = np.zeros_like(input_mask)            
    atlas = load_nii(os.path.join(image_folder,  'tmp', 'MNI_subcortical_mask.nii.gz')).get_data()
    for l in range(1,15):
        
        th_label = input_mask == l
        labels, num_labels = ndimage.label(th_label)
        label_list = np.unique(labels)
        
        # filter candidates by size. Only storing the biggest one
        num_elements_by_lesion = ndimage.labeled_comprehension(np.logical_and(th_label, atlas), labels, label_list, np.sum, float, 0)
        argmax = np.argmax(num_elements_by_lesion)
        
        # assign voxels to output
        current_voxels = np.stack(np.where(labels == argmax), axis =1)
        filtered_mask[current_voxels[:,0], current_voxels[:,1], current_voxels[:,2]] = l

    return filtered_mask


def register_masks(input_mask):
    """
    - Register the MNI subcortical atlas into the T1-w subject space 
    - Input:
    -    input_mask: path to the t1-w input mask used as reference

    - Output:
    -    Elapsed time 

    """
    [image_dir, name] = os.path.split(input_mask)

    # mk a tmp folder to store registered atlases
    try:
        os.mkdir(os.path.join(image_dir, 'tmp'))
    except:
        pass

    s_time = time.time()

    CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
    NIFTIREG_PATH = os.path.join(CURRENT_PATH, 'utils','niftyreg') 
    ATLAS_PATH =  os.path.join(CURRENT_PATH, 'atlases')

    # rigid registration
    if os.path.exists(os.path.join(image_dir, 'tmp','rT1_template.nii.gz')) is False:

        os.system((NIFTIREG_PATH + '/reg_aladin -ref ' + input_mask + 
                   ' -flo ' + os.path.join(ATLAS_PATH, 'T1_template.nii.gz') + 
                   ' -aff ' + os.path.join(image_dir, 'tmp','transf.txt') +
                   ' -res ' + os.path.join(image_dir, 'tmp','rT1_template.nii.gz') + '>  /dev/null'))

    # deformable registration
    if os.path.exists(os.path.join(image_dir, 'tmp','rT1d_template.nii.gz')) is False:
        os.system((NIFTIREG_PATH + '/reg_f3d -ref ' + input_mask + 
                   ' -flo ' + os.path.join(ATLAS_PATH, 'T1_template.nii.gz') + 
                   ' -aff ' + os.path.join(image_dir, 'tmp','transf.txt') +
                   ' -cpp ' + os.path.join(image_dir, 'tmp','transform.nii') +
                   ' -res ' + os.path.join(image_dir, 'tmp','rT1d_template.nii.gz') + '>  /dev/null'))


    # register the atlas back to the image space

    if os.path.exists(os.path.join(image_dir, 'tmp', 'MNI_sub_probabilities.nii.gz')) is False:
        t1 = nib.load(input_mask)
        atlas = nib.load(os.path.join(ATLAS_PATH, 'atlas_subcortical_MNI.nii.gz'))
        s_atlas = np.zeros(t1.get_data().shape + (15,)).astype(np.float32)
        for st in range(15):
            tmp_atlas = nib.Nifti1Image(atlas.get_data()[:,:,:,st], affine = atlas.affine)
            tmp_atlas.to_filename(os.path.join(image_dir, 'tmp','tmp.nii.gz'))
            os.system((NIFTIREG_PATH + '/reg_resample -ref ' + input_mask + 
                       ' -flo ' + os.path.join(image_dir, 'tmp', 'tmp.nii.gz') + 
                       ' -trans ' + os.path.join(image_dir, 'tmp','transform.nii') +
                       ' -res ' + os.path.join(image_dir, 'tmp','r_tmp.nii.gz') + '>  /dev/null'))
            r_tmp_atlas = nib.load(os.path.join(image_dir, 'tmp', 'r_tmp.nii.gz'))
            s_atlas[:,:,:,st] = r_tmp_atlas.get_data().astype(np.float32)

        # save the atlas
        subocortical_atlas = nib.Nifti1Image(s_atlas, affine = t1.affine)
        subocortical_atlas.to_filename(os.path.join(image_dir, 'tmp', 'MNI_sub_probabilities.nii.gz'))

        # generate dilated binary mask (not consider background class = 15)
        mask = np.sum(s_atlas[:,:,:,0:13], axis = 3) > 0
        dilated_mask = ndimage.binary_dilation(mask, iterations = 5)

        binary_atlas_mask = nib.Nifti1Image(dilated_mask.astype('float32'), affine = t1.affine)
        binary_atlas_mask.to_filename(os.path.join(image_dir, 'tmp', 'MNI_subcortical_mask.nii.gz'))

    return time.time() - s_time
    
