###########################################
# Author:     Alfonso Medela              #
# Contact     alfonso@legit.health        #
# Copyright:  Legit.Health                #
# Website:    https://legit.health/       #
###########################################

import glob
import cv2
import numpy as np
import configparser

if __name__ == '__main__':

    # Load configuration file
    root = '../'
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

        path = root_path + dataset_name
        images = glob.glob(path + '*')

        h, w = [], []
        for image in images:
            # Load image
            img = cv2.imread(image)

            # Get shape
            a, b, _ = img.shape
            h.append(a)
            w.append(b)

        h, w = np.asarray(h), np.asarray(w)

        print(dataset_name + ' image statistics:')
        print('width', np.min(w), np.mean(w), np.max(w))
        print('height', np.min(h), np.mean(h), np.max(h))
