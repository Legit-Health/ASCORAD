###########################################
# Author:     Alfonso Medela              #
# Contact     alfonso@legit.health        #
# Copyright:  Legit.Health                #
# Website:    https://legit.health/       #
###########################################

from fastai.vision import *
import glob
import numpy as np
import cv2
import configparser
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, jaccard_score
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

def validate(image_paths, path_label, model_dir, t=0.5):

    '''
    Function to obtain the main metric on lesion segmentation
    :param image_paths: list containing image paths
    :param path_label: path to labels
    :param model_dir: main directory of the models where the folds are saved
    :param t: threshold
    :return: AUC, [ACC, IOU, F1]
    '''

    # Load learner
    learn = load_learner(model_dir)

    metrics = []
    auc_metric = []
    for i_img in tqdm(range(len(image_paths))):

        image_path = image_paths[i_img]
        path_to_annotation = path_label + image_path.split('/')[-1][:-4] + '.png'

        label = cv2.imread(path_to_annotation, -1)
        y = label.reshape(label.shape[0] * label.shape[1])

        img = PIL.Image.open(image_path)
        original_image = np.array(img)
        a = original_image.shape[0]
        b = original_image.shape[1]
        if a > b:
            original_image = cv2.resize(original_image, (b, b))
        else:
            original_image = cv2.resize(original_image, (a, a))

        img = PIL.Image.fromarray(original_image).convert('RGB')
        img = pil2tensor(img, np.float32)
        img = img.div_(255)
        img = Image(img)

        learn.model.eval()
        _, _, mask = learn.predict(img)

        mask = mask.detach().numpy()
        mask = mask[1, :, :, np.newaxis]

        mask_show = cv2.resize(mask, (b, a))
        mask_flatten_int = mask_show.reshape((a * b, mask.shape[-1]))

        mask_flatten = mask_flatten_int.copy()
        mask_flatten_int = mask_flatten_int[:, 0]
        mask_flatten = mask_flatten[:, 0]

        mask_flatten[mask_flatten > t] = 1
        mask_flatten[mask_flatten <= t] = 0

        # IoU
        jac = jaccard_score(y, mask_flatten)

        # F1 score
        f1 = f1_score(y, mask_flatten)

        # Pixel Accuracy
        acc = accuracy_score(y, mask_flatten)

        metrics.append([acc, jac, f1])

        try:
            # AUC only for images with lesion
            auc = roc_auc_score(y, mask_flatten_int)
            auc_metric.append(auc)
        except:
            continue

    auc_metric = np.asarray(auc_metric)
    metrics = np.asarray(metrics)

    # destroy learner
    learn.destroy()
    return np.mean(auc_metric), np.mean(metrics, axis=0)


if __name__ == '__main__':

    root = '../../../'
    SERVABLE_CFG_FILE = root + 'config.ini'
    config = configparser.ConfigParser()
    config.read(SERVABLE_CFG_FILE)

    # Input params defined in config file
    k_folds = int(config['PARAMS']['K_FOLDS'])

    # Get paths from config
    root_path = config['DATA']['DATASET_ROOT_PATH']
    results_path = config['DATA']['RESULTS_ROOT_PATH']

    # Set main path and output path
    path = root_path + 'LegitHealth-AD-FPK-IVI/'
    output_path = results_path + 'lesion-segmentation/exp1/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Set the path to save the models
    model_dir_path = 'MODEL_PATH'

    # Load image paths
    image_paths = glob.glob(path + '*')
    image_paths = np.asarray(image_paths)

    # Paths to ground truth labels
    path_labels = path + 'labels/lesion_segmentation/ground_truth_masks/'

    total_auc = []
    total_metrics = []
    for n_fold in range(k_folds):
        print('Fold ' + str(n_fold))

        # Fold directory
        model_dir = model_dir_path + 'fold-' + str(n_fold) + '/'

        auc_split, metrics_split = validate(image_paths, path_labels, model_dir)

        total_auc.append(auc_split)
        total_metrics.append(metrics_split)

        n_fold += 1

    total_auc = np.asarray(total_auc)
    total_metrics = np.asarray(total_metrics)

    AUC, AUC_std = np.mean(total_auc)*100., np.std(total_auc)*100.
    metrics, metrics_std = np.mean(total_metrics, axis=0)*100., np.std(total_metrics, axis=0)*100.

    df = [AUC, AUC_std,
          metrics[0], metrics_std[0],
          metrics[1], metrics_std[1],
          metrics[2], metrics_std[2]
          ]

    df = np.asarray(df)[np.newaxis, :]

    df = pd.DataFrame(df)
    df.columns = ['AUC',
                  'AUC std',
                  'Px Acc',
                  'Px acc std',
                  'IoU',
                  'IoU std',
                  'F1',
                  'F1 std'
                  ]

    df = df.astype(float)
    df = df.round(2)
    df.to_csv(output_path + 'LegitHealth-AD-FPK-IVI segmentation metrics.csv', index=False)

