import numpy as np
import nibabel as nib
import os
from metrics import HD, DSC
from joblib import Parallel, delayed


def compute_metrics(scan, segmentation, reference):
    """
    compute the multi-class DSC and MHD (modified Hausdorff Distance) between 
    a segmentation and a reference image 
    
    """
    
    GT = nib.load(reference).get_data()
    SEG = nib.load(segmentation).get_data()
    GT[GT == 15] = 0
    labels = np.unique(GT)

    dsc = np.array([DSC(GT == l, SEG == l) for l in range(1, len(labels))])
    mhd = np.array([HD(GT == l, SEG == l, spacing = [1,1,1]) for l in range(1, len(labels))])

    return {scan: [dsc, mhd]}


def compute_stats_for_all_scans(options, mask_name, gt_name, num_cores = 9):
    """
    Compute the DICE and MHD (modified Hausdorff Distance) for all images in 
    the database (running in parallel)

    Input: 
    - options dictionary
    - mask_name: 'segmentation.nii.gz'
    - gt_names: 'gt.nii.gz'
    - num of cores to use 

    Output:
    - A list of results where each element is dictionary['scan_name'] = [DSC, MHD]
    """
    
    list_of_folders = os.listdir(options['folder'])
    list_of_folders.sort(reverse=False)
    
    hd_list = np.array([len(list_of_folders), 14])
    dsc_list = np.array([len(list_of_folders), 14])

    
    # compute stats in parallel
    results = Parallel(n_jobs=num_cores)(delayed(compute_metrics)(f,
                                                                  os.path.join(options['folder'], f, options['model'], 'generated_masks', mask_name),
                                                                  os.path.join(options['folder'], f, gt_name)) for f in list_of_folders)

    return results 

def compute_stats_by_threshold(options, mask_name, gt_name, th_list, num_cores = 9):
    """
    Compute DSC and HD distance for each threshold passed as input
   
    Input: 
    - options dictionary
    - mask_name: 'segmentation.nii.gz'
    - gt_names: 'gt.nii.gz'
    - th_list: list of thresholds to test
    - num of cores to use

    Output:
    - A list of lists where each element contains the DICE and HD values for each image of the dataset for
      the particular threshold, being the second list a dictionary['scan_name'] = [DSC, MHD]

    """

    list_of_folders = os.listdir(options['folder'])
    list_of_folders.sort(reverse=False)

    results_by_threshold = []

    for t in th_list:
        
        # compute stats in parallel
        mask_name_ = mask_name + '_' +  str(t) + '.nii.gz'
        results_by_threshold.append(Parallel(n_jobs=num_cores)(delayed(compute_metrics)(f,
                                                                                        os.path.join(options['folder'], f, options['model'], 'generated_masks', mask_name_),
                                                                                        os.path.join(options['folder'], f, gt_name)) for f in list_of_folders))


    return results_by_threshold




def generate_latex_tables(input_stats, options):
    """
    Prints results by struct in Org-mode table format 
    
    Inputs:
    - input_stats: a list of stats computed with compute_stats_for_all_scans or compute_stats_by_threshold 

    
    """
    mean_stat = np.mean(input_stats, axis = 1)
    std_stat = np.std(input_stats, axis = 1)
    mean_all = np.mean(mean_stat, axis = 1)
    std_all = np.mean(std_stat, axis = 1)

    index = 0
    for st in options['structs']:
        print "| ", st,
        for i in range(len(input_stats)-1):
            print "|",  "{0:.2f}".format(mean_stat[i][index]), '(', "{0:.2f}".format(std_stat[i][index]), ')', 
        print "|",  "{0:.2f}".format(mean_stat[i+1][index]), '(', "{0:.2f}".format(std_stat[i+1][index]), ')'
        index +=1
    # print mean
    print "| mean ", 
    for i in range(len(input_stats)):
        print "|",  "{0:.2f}".format(mean_all[i]), " (", "{0:.2f}".format(std_all[i]), ")",  

        




