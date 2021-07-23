###########################################
# Author:     Alfonso Medela              #
# Contact     alfonso@legit.health        #
# Copyright:  Legit.Health                #
# Website:    https://legit.health/       #
###########################################

# Imports
import glob
import sys
import configparser

# Custom imports
sys.path.append('../../')
from utils.data_processing import ground_truth_generator_segmentation

if __name__ == '__main__':

    # Load configuration file
    root = '../../'
    SERVABLE_CFG_FILE = root + 'config.ini'
    config = configparser.ConfigParser()
    config.read(SERVABLE_CFG_FILE)

    # Get paths
    root_path = config['DATA']['DATASET_ROOT_PATH']
    dataset_names = ['LegitHealth-AD',
                     'LegitHealth-AD-Test',
                     'LegitHealth-AD-FPK-IVI'
                     ]

    for dataset_name in dataset_names:

        # Path to save the GTs
        output_path = root_path + dataset_name + 'labels/lesion_segmentation/ground_truth_masks/'

        # Get filenames
        imgs_path = root_path + dataset_name + 'images/'
        imgs = glob.glob(imgs_path + '*')

        # Path to the annotation masks
        masks_path = root_path + dataset_name + 'labels/lesion_segmentation/masks/'

        # Run main function to generate the ground truth
        ground_truth_generator_segmentation(imgs, masks_path, output_path)



