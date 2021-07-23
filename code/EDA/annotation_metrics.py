###########################################
# Author:     Alfonso Medela              # 
# Contact     alfonso@legit.health        #
# Copyright:  Legit.Health                #
# Website:    https://legit.health/       #
###########################################

# Imports
import cv2
import json
import sys
from tqdm import tqdm
import configparser
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, jaccard_score

# Custom imports
sys.path.append('../')
from utils.metrics import PAR_LH, FAR_LH_mask, realtive_area, unify_masks

def main(mask_path, annotations, visual_signs, number_of_annotators, path_to_vs_cvs, dataset_version):

    '''
    Main function to obtain visual sign intensity and lesion segmentation annotation metrics
    :param mask_path: Path to the masks drawn by the annotators (in PNG format)
    :param annotations: JSON containing visual sign annotations
    :param visual_signs: List of the visual signs evaluated by the annotators
    :param number_of_annotators: The number of annotators
    :param path_to_vs_cvs: Output path for the generated CSVs
    :param dataset_version: Dataset version: 1, 2, 3
    :return: None
    '''

    # Get the number of images from the length of the json
    img_num = len(annotations)

    auc_metric = []
    ls_metrics = []
    vs_results_dataset = []
    area_results_dataset = []
    mask_dataset = []
    for i_img in tqdm(range(1, img_num + 1)):
        filename_no_ext = annotations['img' + str(i_img)]['filename'][:-4]

        # Load masks
        masks = []
        for i_mask in range(1, number_of_annotators + 1):
            mark_path_n = mask_path + filename_no_ext + '_' + str(i_mask) + '.png'
            mask_n = cv2.imread(mark_path_n)
            masks.append(mask_n)

        mask_dataset.append(masks)

        # Compute the average mask
        average_mask = unify_masks(masks)
        average_mask = average_mask / 255
        average_mask = average_mask.reshape(average_mask.shape[0] * average_mask.shape[1])

        # Compute F1, AUC, ACC and IoU metrics at image level
        auc_annotators = []
        metrics_annotators = []
        for i_mask in range(number_of_annotators):
            mask_n = masks[i_mask][:, :, 0]
            mask_n[mask_n > 255/2] = 255
            mask_n[mask_n <= 255/2] = 0
            mask_n = mask_n / 255

            mask_n = mask_n.reshape(mask_n.shape[0] * mask_n.shape[1])
            try:
                # It might fail due to images with all 0's
                auc = roc_auc_score(average_mask, mask_n)
                auc_annotators.append(auc)
            except:
                pass

            # IoU
            jac = jaccard_score(average_mask, mask_n)

            # F1 score
            f1 = f1_score(average_mask, mask_n)

            # Pixel Accuracy
            acc = accuracy_score(average_mask, mask_n)

            metrics_annotators.append([acc, jac, f1])

        try:
            # Skip if some annotator didn't count
            auc_metric.append([auc_annotators[0],
                               auc_annotators[1],
                               auc_annotators[2]
                               ])
            ls_metrics.append([metrics_annotators[0],
                               metrics_annotators[1],
                               metrics_annotators[2]])
        except:
            pass

        # Mask data analysis
        ra_annotators = []
        for i_mask in range(0, number_of_annotators):
            ra = realtive_area(masks[i_mask], t=155)
            ra_annotators.append(ra)

        # Visual sign data analysis
        vs_results = np.empty([len(visual_signs), number_of_annotators])
        for i_vs in range(len(visual_signs)):
            for i_ann in range(1, number_of_annotators + 1):
                vs_n = annotations['img' + str(i_img)]['labeller' + str(i_ann)]['visualSigns'][visual_signs[i_vs]]
                vs_results[i_vs, i_ann - 1] = vs_n

        vs_results_dataset.append(vs_results)
        area_results_dataset.append(ra_annotators)


    vs_results_dataset, area_results_dataset = np.asarray(vs_results_dataset), np.asarray(area_results_dataset)

    # RANDOM - Baseline to understand random values of each metric
    random_dataset = []
    for i in range(1000000):
        vs_results = np.empty([len(visual_signs), number_of_annotators])
        for i_vs in range(len(visual_signs)):
            for i_ann in range(1, number_of_annotators + 1):
                rdn = np.random.randint(0, 4)
                vs_results[i_vs, i_ann - 1] = rdn

        random_dataset.append(vs_results)

    random_dataset = np.asarray(random_dataset)

    # Get statistics - Visual Signs
    # Mean and median
    GT_mean = np.repeat(np.mean(vs_results_dataset, axis=-1)[:, :, np.newaxis], 3, axis=-1)
    GT_median = np.repeat(np.median(vs_results_dataset, axis=-1)[:, :, np.newaxis], 3, axis=-1)

    # Mean and median for random data
    GT_mean_rdn = np.repeat(np.mean(random_dataset, axis=-1)[:, :, np.newaxis], 3, axis=-1)
    GT_median_rdn = np.repeat(np.median(random_dataset, axis=-1)[:, :, np.newaxis], 3, axis=-1)

    # Standard deviation and relative standard deviation
    STD = np.mean(np.std(vs_results_dataset, axis=-1), axis=0)
    RSTD = STD * 100. / 3.

    # Standard deviation and relative standard deviation for random data
    STD_rdn = np.mean(np.std(random_dataset, axis=-1), axis=0)
    RSTD_rdn = STD_rdn * 100. / 3.

    # Mean absolute error and relative mean absolute error calculation
    MAE_mean = np.mean(np.mean(np.abs(vs_results_dataset - GT_mean), axis=-1), axis=0)
    RMAE_mean = MAE_mean * 100. / 3.

    # Mean absolute error and relative mean absolute error calculation for random data
    MAE_mean_rdn = np.mean(np.mean(np.abs(random_dataset - GT_mean_rdn), axis=-1), axis=0)
    RMAE_mean_rdn = MAE_mean_rdn * 100. / 3.

    # Mean absolute error and relative mean absolute error calculation using the median
    MAE_median = np.mean(np.mean(np.abs(vs_results_dataset - GT_median), axis=-1), axis=0)
    RMAE_median = MAE_median * 100. / 3.

    # Mean absolute error and relative mean absolute error calculation using the median for random data
    MAE_median_rdn = np.mean(np.mean(np.abs(random_dataset - GT_median_rdn), axis=-1), axis=0)
    RMAE_median_rdn = MAE_median_rdn * 100. / 3.

    # Full Agreement Rate calculation
    std = np.std(vs_results_dataset, axis=-1)
    std[std < 0] = 1
    std[std > 0] = 1
    FAR = (len(std) - np.sum(std, axis=0)) * 100 / len(std)

    # Full Agreement Rate calculation for random data
    std_rdn = np.std(random_dataset, axis=-1)
    std_rdn[std_rdn < 0] = 1
    std_rdn[std_rdn > 0] = 1
    FAR_rdn = (len(std_rdn) - np.sum(std_rdn, axis=0)) * 100 / len(std_rdn)

    # Partial Agreement Rate calculation
    PAR1, PAR2 = PAR_LH(vs_results_dataset, visual_signs, number_of_annotators)
    PAR1_rdn, PAR2_rdn = PAR_LH(random_dataset, visual_signs, number_of_annotators)

    # Save as csv
    df = []
    df.append(visual_signs)
    df.append(STD)
    df.append(STD_rdn)
    df.append(RSTD)
    df.append(RSTD_rdn)
    df.append(MAE_mean)
    df.append(MAE_mean_rdn)
    df.append(RMAE_mean)
    df.append(RMAE_mean_rdn)
    df.append(MAE_median)
    df.append(MAE_median_rdn)
    df.append(RMAE_median)
    df.append(RMAE_median_rdn)
    df.append(FAR)
    df.append(FAR_rdn)
    df.append(PAR1)
    df.append(PAR1_rdn)
    df.append(PAR2)
    df.append(PAR2_rdn)

    df = np.asarray(df)
    df = np.transpose(df)
    df = pd.DataFrame(df)
    df.columns = ['Visual signs',
                  'STD', 'STD Random',
                  'RSTD', 'RSTD random',
                  'MAE mean', 'MAE mean random',
                  'RMAE mean', 'RMAE mean random',
                  'MAE median', 'MAE median random',
                  'RMAE median', 'RMAE median random',
                  'FAR', 'FAR random',
                  'PAR 1', 'PAR 1 random',
                  'PAR 2', 'PAR 2 random'
                  ]

    # Convert values to float and round to 2 decimals
    df[df.columns[1:]] = df[df.columns[1:]].astype(float)
    df[df.columns[1:]] = df[df.columns[1:]].round(2)

    # Save the results
    df.to_csv(path_to_vs_cvs + dataset_version + ' visual sign metrics.csv', index=False)

    # Get statistics for lesion surface segmentation
    # Standard deviation
    STD_sf = np.mean(np.std(area_results_dataset, axis=-1))

    # AUC, F1, ACC, IoU
    auc_metric = np.asarray(auc_metric)
    AUC = np.mean(auc_metric)*100.

    ls_metrics = np.asarray(ls_metrics)
    ls_metrics = np.mean(np.mean(ls_metrics, axis=-1), axis=0) * 100.

    df_ls = []
    df_ls.append(STD_sf)
    df_ls.append(AUC)
    df_ls.append(ls_metrics[0])
    df_ls.append(ls_metrics[1])
    df_ls.append(ls_metrics[2])

    df_ls = np.asarray(df_ls)[np.newaxis, :]
    df_ls = pd.DataFrame(df_ls)
    df_ls.columns = ['RSD', 'AUC', 'ACC', 'IoU', 'F1']

    # Convert values to float and round to 2 decimals
    df_ls = df_ls.astype(float)
    df_ls = df_ls.round(2)

    # Save the results
    df_ls.to_csv(path_to_vs_cvs + dataset_version + ' lesion surface metrics.csv', index=False)


if __name__ == '__main__':

    # Load configuration file
    root = '../'
    SERVABLE_CFG_FILE = root + 'config.ini'
    config = configparser.ConfigParser()
    config.read(SERVABLE_CFG_FILE)

    # Load the number of annotators
    number_of_annotators = int(config['ANNOTATION']['NUMBER_OF_ANNOTATORS'])

    # Load visual signs
    vs_num = int(config['ANNOTATION']['VISUAL_SIGN_NUMBER'])
    visual_signs = []
    for i_vs in range(vs_num):
        vs = config['ANNOTATION']['VISUAL_SIGN_' + str(i_vs) + '']
        visual_signs.append(vs)

    # Get paths
    root_path = config['DATA']['DATASET_ROOT_PATH']
    results_path = config['DATA']['RESULTS_ROOT_PATH']
    dataset_names = ['LegitHealth-AD',
                     'LegitHealth-AD-Test',
                     'LegitHealth-AD-FPK-IVI'
                     ]

    for dataset_name in dataset_names:

        # Load annotation data
        json_path = root_path + dataset_name + 'labels/visual_sign_assessment/visual_sign_intensities.json'
        with open(json_path) as json_file:
            annotations = json.load(json_file)

        # Set paths for the images and segmentation masks
        imgs_path = root_path + dataset_name + '/images/'
        mask_path = root_path + dataset_name + '/labels/lesion_segmentation/masks/'

        # Path to save the results
        path_to_vs_cvs = root + results_path + 'annotation/'

        # Run main fuction
        main(mask_path, annotations, visual_signs, number_of_annotators, path_to_vs_cvs, dataset_name)




