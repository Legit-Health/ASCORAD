###########################################
# Author:     Alfonso Medela              #
# Contact     alfonso@legit.health        #
# Copyright:  Legit.Health                #
# Website:    https://legit.health/       #
###########################################

# Imports
import numpy as np

def PAR_LH(vs_results_dataset, visual_signs, number_of_annotators):

    '''
    Partial Agreement Rate Metric
    :param vs_results_dataset: visual sign results [image x signs (6) x annotators (3)]
    :param visual_signs: visual sign list
    :param number_of_annotators: (3)
    :return: Partial Agreement 1 and 2 metrics
    '''

    PAR1_metric = [0 for i in range(len(visual_signs))]
    PAR2_metric = [0 for i in range(len(visual_signs))]

    for i_img in range(vs_results_dataset.shape[0]):
        for i_vs in range(vs_results_dataset.shape[1]):
            labels = []
            for i_ann in range(0, number_of_annotators):
                labels.append(vs_results_dataset[i_img, i_vs, i_ann])

            if labels[0] == labels[1]:
                PAR2_metric[i_vs] += 1

                dif = abs(labels[0] - labels[2])
                if dif <= 1.0:
                    PAR1_metric[i_vs] += 1

            elif labels[1] == labels[2]:
                PAR2_metric[i_vs] += 1

                dif = abs(labels[0] - labels[2])
                if dif <= 1.0:
                    PAR1_metric[i_vs] += 1

            elif labels[0] == labels[2]:
                PAR2_metric[i_vs] += 1

                dif = abs(labels[1] - labels[2])
                if dif <= 1.0:
                    PAR1_metric[i_vs] += 1

    PAR1_metric = np.asarray(PAR1_metric) * 100 / vs_results_dataset.shape[0]
    PAR2_metric = np.asarray(PAR2_metric) * 100 / vs_results_dataset.shape[0]
    return PAR1_metric, PAR2_metric

def FAR_LH_mask(mask_dataset, number_of_annotators):

    '''
    Full Agreement Rate Metric for segmentation mask
    :param mask_dataset: masks
    :param number_of_annotators: (3)
    :return: Full Agreement Rate metric
    '''

    PAR2_metric = 0
    total_px = 0

    for i_img in range(mask_dataset.shape[0]):
        labels = []
        for i_ann in range(0, number_of_annotators):
            labels.append(np.ndarray.flatten(mask_dataset[i_img, i_ann]))

        labels = np.asarray(labels)
        for i_px in range(labels.shape[-1]):
            if labels[0, i_px] == labels[1, i_px] == labels[2, i_px]:
                PAR2_metric += 1
                continue

        total_px += labels.shape[-1]

    PAR2_metric = PAR2_metric * 100 / total_px
    return PAR2_metric

def realtive_area(mask, t=155):

    '''
    Fuction that get the percentage of lesion respect to the total number of pixels
    :param mask: input mask
    :param t: threshold parameter to determine if lesion or not
    :return: Relative area (%)
    '''

    mask[mask > t] = 255
    mask[mask <= t] = 0

    no_lesion_px_count = np.unique(mask, return_counts=True)[-1][0]

    try:
        lesion_px_count = np.unique(mask, return_counts=True)[-1][1]
        total_px = lesion_px_count + no_lesion_px_count

        # Relative Area Calculation
        relative_area = lesion_px_count * 100 / total_px
    except:
        # No lesion was found
        relative_area = 0

    return relative_area

def unify_masks(masks, t=155):

    '''
    Get the GT mask by averaging
    :param masks: input masks (3)
    :param t: threshold parameter
    :return: average mask
    '''

    merge_masks = np.concatenate((masks[0][:, :, :1], masks[1][:, :, :1], masks[2][:, :, :1]), axis=-1)
    unified_mask = np.mean(merge_masks, axis=-1)
    unified_mask[unified_mask > t] = 255
    unified_mask[unified_mask <= t] = 0
    return unified_mask