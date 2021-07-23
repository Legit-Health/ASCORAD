###########################################
# Author:     Alfonso Medela              #
# Contact     alfonso@legit.health        #
# Copyright:  Legit.Health                #
# Website:    https://legit.health/       #
###########################################

from fastai.vision import *
import glob
import time
import warnings
import configparser
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

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

    # Use the largest dataset to get more precise results on execution time
    path = root_path + dataset_names[0]
    img_paths = glob.glob(path + '*')

    A_learner_path = 'LEARNER_A_PATH'
    B_learner_path = 'LEARNER_B_PATH'

    exec_time = []
    for img_path in img_paths:

        t0 = time.time()

        # Open image
        img = open_image(img_path)

        # Learner paths
        ta_0 = time.time()
        A_learner = load_learner(A_learner_path)
        _ = A_learner.predict(img)
        ta = time.time() - t0

        B_learner = load_learner(B_learner_path)
        _ = B_learner.predict(img)
        tb = time.time() - t0 - (ta - ta_0)

        t_total = time.time() - t0
        exec_time.append([ta, tb, t_total])

        # Delete learners
        A_learner.destroy()
        B_learner.destroy()

    exec_time = np.asarray(exec_time)
    mean_exec_time = np.mean(exec_time, axis=0)
    print(mean_exec_time)


