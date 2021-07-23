###########################################
# Author:     Alfonso Medela              #
# Contact     alfonso@legit.health        #
# Copyright:  Legit.Health                #
# Website:    https://legit.health/       #
###########################################

# Imports
from fastai.vision import *
from sklearn.model_selection import KFold
import glob
import numpy as np
import configparser
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

# Custom imports
sys.path.append('../../')
from utils.model_export import custom_export

def training_loop(path_img, model_dir_path, all_images_paths, codes, size, bs, get_y_fn, k_folds, seed, lr_find=False):

    '''
    Training function for image segmentation
    :param path_img: path to images
    :param model_dir_path: path to save models
    :param all_images_paths: list of all image paths
    :param codes: names of the classes
    :param size: image size
    :param bs: batch size
    :param get_y_fn: function to get label names
    :param k_folds: number of folds
    :param seed: random seed
    :param lr_find: True if we want to find the optimal LR and false to directly train
    :return: None
    '''

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

    n_fold = 0
    for train_index, test_index in kf.split(all_images_paths):
        print('Fold ' + str(n_fold))

        train_list, val_list = all_images_paths[train_index], all_images_paths[test_index]
        val_list_final = []
        for i_l in range(len(val_list)):
            val_list_final.append(val_list[i_l].split('/')[-1])

        model_dir = model_dir_path + 'fold-' + str(n_fold) + '/'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # create valid.txt
        src = (SegmentationItemList.from_folder(path_img)
               .split_by_files(val_list)
               .label_from_func(get_y_fn, classes=codes))

        data = (src.transform(get_transforms(max_zoom=1.3, max_lighting=0.4, max_warp=0.4, p_affine=1., p_lighting=1.),
                              size=size, tfm_y=True)
                .databunch(bs=bs, num_workers=0)
                .normalize(imagenet_stats))

        # Define metrics
        metrics = [partial(dice, iou=True), dice]

        wd = 1e-2
        learn = unet_learner(data, models.resnet34, metrics=metrics, wd=wd)
        learn.model_dir = model_dir

        lr_def = 0
        if lr_find:
            learn.lr_find()
            fig = learn.recorder.plot(return_fig=True)
            fig.savefig('lr_figure_freezed_mixup.png')
        else:
            print('Training...')
            if lr_def == 0:
                lr = 1e-3
            else:
                lr = lr_def

            learn.fit_one_cycle(10, slice(lr), pct_start=0.9)
            learn.save('stage_1')
            print('Head training finished')

            learn.unfreeze()
            lrs = slice(lr / 400, lr / 4)
            learn.fit_one_cycle(12, lrs, pct_start=0.8)
            learn.save('stage_2')

            # Export model
            custom_export(model_dir, learn)

            # destroy learner
            learn.destroy()

        print('Training ' + 'fold ' + str(n_fold) + ' finished')
        n_fold += 1


if __name__ == '__main__':

    # Load configuration file
    root = '../../'
    SERVABLE_CFG_FILE = root + 'config.ini'
    config = configparser.ConfigParser()
    config.read(SERVABLE_CFG_FILE)

    # Input params defined in config file
    seed = int(config['PARAMS']['SEED'])
    device = int(config['PARAMS']['DEVICE'])
    bs = int(config['PARAMS']['BS_SEG'])
    size = int(config['PARAMS']['IMG_SIZE'])
    k_folds = int(config['PARAMS']['K_FOLDS'])

    # Classes
    category_0 = config['TAXONOMY']['CATEGORY_0']
    category_1 = config['TAXONOMY']['CATEGORY_1']
    codes = [category_0, category_1]

    # Set the path to save the models
    model_dir_path_1 = 'MODEL_PATH_EXP1'
    model_dir_path_2 = 'MODEL_PATH_EXP2'

    # Set device
    torch.cuda.set_device(device)

    # Experiment 1
    # Get paths
    root_path = config['DATA']['DATASET_ROOT_PATH']
    path = root_path + 'LegitHealth-AD/'
    path_img = path + 'images/'
    path_label = path + 'labels/lesion_segmentation/ground_truth_masks/'
    get_y_fn = lambda x: path_label + f'{x.stem}.png'

    # Read image paths for K-fold
    all_images_paths = glob.glob(path_img + '*')
    all_images_paths = np.asarray(all_images_paths)

    # Training loop using K-fold strategy
    training_loop(path_img, model_dir_path_1, all_images_paths, codes, size, bs, get_y_fn, k_folds, seed)

    # Experiment 2
    # Get paths
    path = root_path + 'LegitHealth-V1-V2-V3'
    path_img = path + 'images/'
    path_label = path + 'labels/lesion_segmentation/ground_truth_masks/'
    get_y_fn = lambda x: path_label + f'{x.stem}.png'

    # Load only V3 images for K-fold
    path = root_path + 'LegitHealth-AD-FPK-IVI/'
    v3_image_paths = glob.glob(path + '*')
    v3_image_paths = np.asarray(v3_image_paths)

    training_loop(path_img, model_dir_path_2, v3_image_paths, codes, size, bs, get_y_fn, k_folds, seed)















