# --------------------------------------------------
# load options for CNN training / testing
#
# Options are loaded from a configuration file
#
# --------------------------------------------------

import ConfigParser, os


def load_options(user_config):
    """
    map options from user input into the default config 
    """
    
    sections = user_config.sections()

    # --------------------------------------------------
    # options
    # --------------------------------------------------
    options = {}

    # experiment name (where trained weights are)
    options['experiment'] = user_config.get('model', 'name')
    options['train_folder'] = user_config.get('database', 'train_folder')
    options['test_folder'] = user_config.get('database', 'inference_folder')
    options['output_folder'] = ''
    options['current_scan'] = ''
    options['t1_name'] = user_config.get('database', 't1_name')
    options['roi_name'] = user_config.get('database', 'roi_name')
    options['out_name'] = 'out_seg.nii.gz'
    options['save_tmp'] = user_config.get('database', 'save_tmp')
    exp_folder = None 

    # net options
    options['mode'] = user_config.get('model', 'mode')
    options['patch_size'] = ([user_config.getint('model', 'patch_size'),
                             user_config.getint('model', 'patch_size')])
    options['weight_paths'] = None
    options['train_split'] = user_config.getfloat('model', 'train_split')
    options['max_epochs'] = user_config.getint('model', 'max_epochs')
    options['patience'] = user_config.getint('model', 'patience')
    options['batch_size'] = user_config.getint('model', 'batch_size')
    options['test_batch_size'] = user_config.getint('model', 'test_batch_size')
    options['net_verbose'] = user_config.getint('model', 'net_verbose')
    options['load_weights'] = user_config.get('model', 'load_weights')
    options['randomize_train'] = True
    options['debug'] = user_config.get('model', 'debug')
    options['out_probabilities'] = user_config.get('model', 'out_probabilities')
    options['post_process'] = user_config.get('model', 'post_process')
    options['crop'] = user_config.get('model', 'speedup_segmentation')
    
    return options 


def print_options(options):
    """ 
    print options 
    """

    print "--------------------------------------------------"
    print " "
    keys = options.keys()
    for k in keys:
        print k, ':', options[k]
    print "--------------------------------------------------"
    
    
    
