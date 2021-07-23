###########################################
# Author:     Alfonso Medela              #
# Contact     alfonso@legit.health        #
# Copyright:  Legit.Health                #
# Website:    https://legit.health/       #
###########################################

# Imports
import glob
import configparser
from sklearn.model_selection import KFold
from fastai.vision import *
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

# Custom imports
sys.path.append('../../ ')
from utils.multihead_network import CategoryListAdapted, Network, MultipleHeadCrossEntropy, FMAPE
from utils.data_processing import ground_truth_generator, LegitDataloader

def train_loop(image_paths, main_model_dir, path, labels_path, num_classes, bs, sz, k_folds, gt_mode, seed, visual_signs):

    '''
    Training function
    :param image_paths: path to all the filenames
    :param main_model_dir: Directory to save the models
    :param path: Path to image folder
    :param labels_path: Path to label folder
    :param num_classes: Number of total categories
    :param bs: batch size
    :param sz: image size (sz, sz)
    :param k_folds: Number of folds
    :return:
    '''

    # K folds
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

    n_fold = 0
    for train_index, test_index in kf.split(image_paths):
        train_list, val_list = image_paths[train_index], image_paths[test_index]

        # validation list
        val_list_final = []
        for i_l in range(len(val_list)):
            val_list_final.append(val_list[i_l].split('/')[-1])

        tfms = get_transforms()
        data = (ImageList.from_folder(path)
                .split_by_files(val_list_final)
                .label_from_folder()
                .transform(tfms, size=sz, padding_mode='reflection')
                .databunch(num_workers=0, bs=bs)
                .normalize(imagenet_stats)
                )

        multihead_efficientnet = Network(num_classes, device)
        learn = Learner(data, multihead_efficientnet, callback_fns=FMAPE)

        # Load and preprocess dataset
        learn, num_classes = LegitDataloader(learn, labels_path, visual_signs, label_max=3, max_value=num_classes-1, gt_mode=gt_mode)

        # create layer groups
        list_layers = [
            learn.model.model,
            learn.model.l1,
            learn.model.l2,
            learn.model.l3,
            learn.model.l4,
            learn.model.l5,
            learn.model.l6
        ]
        learn.split(list_layers)

        learn.loss_func = MultipleHeadCrossEntropy(device)

        # Create if the folder doesnt exist
        model_dir = main_model_dir + 'fold-' + str(n_fold) + '/'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        learn.model_dir = model_dir

        learn.freeze_to(1)
        learn.fit_one_cycle(20, slice(1e-3))
        learn.save('stage1_weights')

        learn.unfreeze()
        learn.fit_one_cycle(40, slice(1e-4))
        learn.save('stage2_weights')

        learn.fit_one_cycle(10, slice(1e-4))
        learn.save('stage3_weights')

        learn.fit_one_cycle(8, slice(5e-5))
        learn.save('stage4_weights')

        learn.fit_one_cycle(6, slice(5e-5))
        learn.save('stage5_weights')

        # Destroy learner
        learn.destroy()

        n_fold += 1
        print('DONE')


if __name__ == '__main__':

    SERVABLE_CFG_FILE = '../../../config.ini'
    config = configparser.ConfigParser()
    config.read(SERVABLE_CFG_FILE)

    # Input params defined in config file
    seed = int(config['PARAMS']['SEED'])
    device = int(config['PARAMS']['DEVICE'])
    bs = int(config['PARAMS']['BS_CLF'])
    sz = int(config['PARAMS']['IMG_SIZE'])
    k_folds = int(config['PARAMS']['K_FOLDS'])

    # Load visual signs
    vs_num = int(config['ANNOTATION']['VISUAL_SIGN_NUMBER'])
    visual_signs = []
    for i_vs in range(vs_num):
        vs = config['ANNOTATION']['VISUAL_SIGN_' + str(i_vs) + '']
        visual_signs.append(vs)

    # Set torch device
    torch.cuda.set_device(device)

    # Experiment 1
    # Get paths
    root_path = config['DATA']['DATASET_ROOT_PATH']
    root = root_path + 'LegitHealth-AD/'
    path = root + 'images/'
    labels_path = root + 'labels/visual_sign_assessment/visual_sign_intensities.json'

    model_dir_exp1 = 'MODEL_DIR_EXP1'
    model_dir_exp2 = 'MODEL_DIR_EXP2'

    # Load filenames
    image_paths = glob.glob(path + '*')
    image_paths = np.asarray(image_paths)

    gt_modes = ['mean', 'median']
    num_classes_vect = [4, 11, 101]
    for num_classes in num_classes_vect:
        for gt_mode in gt_modes:

            # Model dir
            main_model_dir = model_dir_exp1 + '/range0-' + str(num_classes - 1) + '/' + gt_mode + '/'

            train_loop(image_paths, main_model_dir, path, labels_path, num_classes, bs, sz, k_folds, gt_mode, seed, visual_signs)

    # Experiment 2
    k_folds = 3
    gt_mode = 'median'
    num_classes = 101

    # Paths to v1, v2, v3 data
    root = root_path + 'LegitHealth-V1-V2-V3/'
    path = root + 'images/'
    labels_path = root + 'labels/visual_sign_assessment/visual_sign_intensities.json'

    # Load only V3 images for K-fold
    path = root_path + 'LegitHealth-AD-FPK-IVI/'
    v3_image_paths = glob.glob(path + '*')
    v3_image_paths = np.asarray(v3_image_paths)

    main_model_dir = model_dir_exp2

    train_loop(v3_image_paths, main_model_dir, path, labels_path, num_classes, bs, sz, k_folds, gt_mode, seed, visual_signs)








