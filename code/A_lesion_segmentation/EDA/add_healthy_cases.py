###########################################
# Author:     Alfonso Medela              #
# Contact     alfonso@legit.health        #
# Copyright:  Legit.Health                #
# Website:    https://legit.health/       #
###########################################

# Imports
import glob
import cv2
import shutil
from sklearn.utils import shuffle
import numpy as np
import configparser
from tqdm import tqdm

def move_imgs(healthy_ds_path, dst_path, n_healthy):

    '''
    Function to add the desired number of healthy skin images to the official AD datasets
    :param healthy_ds_path: path to the healthy images
    :param dst_path: Output path where images/ and labels/ folders are located
    :param n_healthy: Number of healthy images to include in the dataset
    :return: None
    '''

    healthy_images = glob.glob(healthy_ds_path + '*')

    # Shuffle and get the desired number of images
    original_img = shuffle(healthy_images)
    original_img = original_img[:n_healthy]

    for i_img in tqdm(range(len(original_img))):

        # Copy the image to the new folder
        src = original_img[i_img]
        dst = dst_path + 'images/' + original_img[i_img].split('/')[-1]
        shutil.copy(src, dst)

        # Load image
        img = cv2.imread(original_img[i_img])

        # Get shape
        shape = img.shape
        a, b = shape[0], shape[1]

        # Create a black mask (all 0's)
        black_mask = np.zeros([a, b])

        # Save mask as a png
        img_name = original_img[i_img].split('/')[-1].split('.')[0]
        cv2.imwrite(dst_path + 'labels/' + img_name + '.png', black_mask)

if __name__ == '__main__':

    # Load configuration file
    root = '../../'
    SERVABLE_CFG_FILE = root + 'config.ini'
    config = configparser.ConfigParser()
    config.read(SERVABLE_CFG_FILE)

    # Get paths
    root_path = config['DATA']['DATASET_ROOT_PATH']
    healthy_ds_path = root_path + 'LegitHealth-HealthySkin/'

    # Run main function
    move_imgs(healthy_ds_path, root_path, n_healthy = 300)

