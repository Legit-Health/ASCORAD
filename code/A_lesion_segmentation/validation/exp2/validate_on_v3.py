###########################################
# Author:     Alfonso Medela              #
# Contact     alfonso@legit.health        #
# Copyright:  Legit.Health                #
# Website:    https://legit.health/       #
###########################################

from fastai.vision import *
from sklearn.model_selection import KFold
import glob
import configparser
import numpy as np
import cv2
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, jaccard_score
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

def validate(path_img, model_dir, val_list, codes, size, bs, get_y_fn, t=0.5):

    src = (SegmentationItemList.from_folder(path_img)
           .split_by_files(val_list)
           .label_from_func(get_y_fn, classes=codes))

    data = (src.transform(get_transforms(max_zoom=1.3, max_lighting=0.4, max_warp=0.4, p_affine=1., p_lighting=1.),
                          size=size, tfm_y=True)
            .databunch(bs=bs, num_workers=0)
            .normalize(imagenet_stats))

    # define metrics
    metrics = [partial(dice, iou=True), dice]

    wd = 1e-2
    learn = unet_learner(data, models.resnet34, metrics=metrics, wd=wd)
    learn.model_dir = model_dir

    learn.load('stage_2')

    metrics = []
    auc_metric = []
    for i_img in range(len(learn.data.valid_ds.x.items)):

        image_path = str(learn.data.valid_ds.x.items[i_img])
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
    seed = int(config['PARAMS']['SEED'])
    device = int(config['PARAMS']['DEVICE'])
    bs = int(config['PARAMS']['BS_SEG'])
    size = int(config['PARAMS']['IMG_SIZE'])
    k_folds = 3

    # Classes
    category_0 = config['TAXONOMY']['CATEGORY_0']
    category_1 = config['TAXONOMY']['CATEGORY_1']
    codes = [category_0, category_1]

    # Set device
    torch.cuda.set_device(device)

    # Load output path
    output_path = root + '/data/results/lesion-segmentation/exp2/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Get paths from config
    root_path = config['DATA']['DATASET_ROOT_PATH']
    path = root_path + 'LegitHealth-V1-V2-V3/'

    model_dir_path = 'MODEL_DIR_EXP2'

    path_img = path + 'images/'
    path_label = path + 'labels/'
    get_y_fn = lambda x: path_label + f'{x.stem}.png'

    # 3-fold crossvalidation only with v3 data
    path = root_path + 'LegitHealth-AD-FPK-IVI/'
    v3_image_paths = glob.glob(path + '*')
    v3_image_paths = np.asarray(v3_image_paths)

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

    n_fold = 0
    total_auc = []
    total_metrics = []
    for train_index, test_index in kf.split(v3_image_paths):
        # Fold number
        print('Fold ' + str(n_fold))

        # Get train and val splits
        train_list, val_list = v3_image_paths[train_index], v3_image_paths[test_index]
        val_list_final = []
        for i_l in range(len(val_list)):
            val_list_final.append(val_list[i_l].split('/')[-1])

        # Model directory
        model_dir = model_dir_path + 'fold-' + str(n_fold) + '/'

        # Run validation function
        auc_split, metrics_split = validate(path_img, model_dir, val_list_final, codes, size, bs, get_y_fn)

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

