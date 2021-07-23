###########################################
# Author:     Alfonso Medela              #
# Contact     alfonso@legit.health        #
# Copyright:  Legit.Health                #
# Website:    https://legit.health/       #
###########################################

# Imports
import glob
from sklearn.model_selection import KFold
from fastai.vision import *
import configparser
import warnings
from sklearn.metrics import mean_absolute_error
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

# Custom imports
sys.path.append('../../../../')
from utils.multihead_network import CategoryListAdapted, Network, MultipleHeadCrossEntropy, FMAPE, softmax
from utils.data_processing import ground_truth_generator, LegitDataloader

def validate(image_paths, main_model_dir, path, labels_path, num_classes, bs, sz, k_folds, gt_mode, seed, visual_signs):

    '''
    Validation function
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
    #kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    kf = KFold(n_splits=k_folds)

    n_fold = 0
    metrics = []
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
        learn.model_dir = model_dir
        learn.load('stage5_weights')

        # Compute metrics
        class_list = learn.data.classes

        # Predictions
        y_pred_1_list = [[] for i in range(len(visual_signs))]
        y_pred_2_list = [[] for i in range(len(visual_signs))]

        # Labels
        y_real_list = [[] for i in range(len(visual_signs))]

        val_x = learn.data.valid_ds.x.items
        val_y = learn.data.valid_ds.y
        for i in range(len(val_x)):

            val_y_i = np.asarray(val_y[i]).astype(int)

            img = open_image(str(val_x[i]))
            pred = learn.predict(img)

            pred_vs_score = [0 for l in range(len(visual_signs))]

            for i_vs in range(len(visual_signs)):
                res_out = pred[-1][i_vs].detach().numpy()
                res = softmax(res_out)
                for j_vs in range(len(class_list)):
                    pred_vs_score[i_vs] += res[j_vs] * class_list[j_vs]

                # Method 1
                y_pred_1 = int(class_list[np.argmax(res_out)])
                y_pred_1_list[i_vs].append(y_pred_1)

                # Method 2
                y_pred_2_list[i_vs].append(pred_vs_score[i_vs])

                # GT
                y_real_list[i_vs].append(val_y_i[i_vs])


        y_pred_1_list, y_pred_2_list, y_real_list = np.asarray(y_pred_1_list), np.asarray(y_pred_2_list), np.asarray(y_real_list)

        range_min = np.min(class_list)
        range_max = np.max(class_list)

        MAE_1 = []
        for i_vs in range(len(visual_signs)):
            mae_vs = mean_absolute_error(y_real_list[i_vs], y_pred_1_list[i_vs])
            rmae_vs = (mae_vs - range_min) * 100 / (range_max+range_min)
            MAE_1.append(rmae_vs)
        MAE_1 = np.asarray(MAE_1)

        MAE_2 = []
        for i_vs in range(len(visual_signs)):
            mae_vs = mean_absolute_error(y_real_list[i_vs], y_pred_2_list[i_vs])
            rmae_vs = (mae_vs - range_min) * 100 / (range_max + range_min)
            MAE_2.append(rmae_vs)
        MAE_2 = np.asarray(MAE_2)

        metrics.append([MAE_1, MAE_2])

        # Destroy learner
        learn.destroy()

        n_fold += 1

    metrics = np.asarray(metrics)
    metrics = np.mean(metrics, axis=0)
    return metrics


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

    # Set device
    torch.cuda.set_device(device)

    # Paths
    root_path = config['DATA']['DATASET_ROOT_PATH']
    root = root_path + 'LegitHealth-AD/'
    path = root + 'images/'
    labels_path = root + 'labels/visual_sign_assessment/visual_sign_intensities.json'

    # Load filenames
    image_paths = glob.glob(path + '*')
    image_paths = np.asarray(image_paths)

    model_dir_exp1 = 'MODEL_DIR_EXP1'

    gt_modes = ['mean', 'median']
    num_classes_vect = [4, 11, 101]
    for num_classes in num_classes_vect:
        for gt_mode in gt_modes:

            # Model dir
            main_model_dir = model_dir_exp1 + '/range0-' + str(num_classes - 1) + '/' + gt_mode + '/'

            # Change gt_mode to mean (validation GT)
            gt_mode = 'mean'
            results = validate(image_paths, main_model_dir, path, labels_path, num_classes, bs, sz, k_folds, gt_mode, seed, visual_signs)
            print(np.mean(results, axis=1))










