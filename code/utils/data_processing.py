###########################################
# Author:     Alfonso Medela              #
# Contact     alfonso@legit.health        #
# Copyright:  Legit.Health                #
# Website:    https://legit.health/       #
###########################################

# Imports
from fastai.vision import *
import cv2
from tqdm import tqdm

# Custom imports
from utils.multihead_network  import CategoryListAdapted

def generate_avg_masks(img_path, mask_path):

    '''
    Function to create average masks
    :param img_path: input image path
    :param mask_path: input mask path
    :return: [mask 1, mask 2, mask 3], average mask
    '''

    img_name = img_path.split('/')[-1].split('.')[0]

    masks = []
    for i_mask in range(1, 4):
        i_mask_path = mask_path + img_name + '_' + str(i_mask) + '.png'
        mask_img = cv2.imread(i_mask_path)
        masks.append(mask_img)

    mask_merge = np.concatenate((masks[0][:, :, :1], masks[1][:, :, :1], masks[2][:, :, :1]), axis=-1)
    mask_mean = np.mean(mask_merge, axis=-1)
    mask_mean[mask_mean > 155] = 255
    mask_mean[mask_mean <= 155] = 0

    mask_mean = mask_mean * 1.0 / 255.

    for i_mask in range(len(masks)):
        masks[i_mask] = masks[i_mask][:, :, 0]
        masks[i_mask][masks[i_mask] > 155] = 255
        masks[i_mask][masks[i_mask] <= 155] = 0
        masks[i_mask] = masks[i_mask] * 1.0 / 255.
    return masks, mask_mean

def ground_truth_generator_segmentation(imgs, mask_path, output_path):

    '''
    Main function that generates the ground truths for lesion segmentation by combining the masks drawn by the annotators
    :param imgs: list of image paths
    :param mask_path: path to the masks, where _n.png is the mask drawn by the n-th annotator
    :param output_path: path to save the masks
    :return: None
    '''

    for i_img in tqdm(range(len(imgs))):
        img_name = imgs[i_img].split('/')[-1].split('.')[0]

        mask_1 = mask_path + img_name + '_1.png'
        mask_2 = mask_path + img_name + '_2.png'
        mask_3 = mask_path + img_name + '_3.png'

        mask_1_open = cv2.imread(mask_1)
        mask_2_open = cv2.imread(mask_2)
        mask_3_open = cv2.imread(mask_3)

        mask_merge = np.concatenate((mask_1_open[:, :, :1], mask_2_open[:, :, :1], mask_3_open[:, :, :1]), axis=-1)
        mask_mean = np.mean(mask_merge, axis=-1)
        mask_mean[mask_mean > 155] = 255
        mask_mean[mask_mean <= 155] = 0

        mask_mean = mask_mean * 1.0 / 255.

        # Save mask
        cv2.imwrite(output_path + img_name + '.png', mask_mean)

def ground_truth_generator(json_data, visual_sign, max_value, label_max, gt_mode):

    '''
    Ground truth generator for visual sign intensities
    :param json_data: Visual sign intensity annotations
    :param visual_sign: visual sign list
    :param max_value: maximum value of the desired range
    :param label_max: maximum value according to SCORAD (3)
    :param gt_mode: median or mean
    :return:
    '''

    # Load annotations
    vs1 = json_data['labeller1']['visualSigns'][visual_sign]
    vs2 = json_data['labeller2']['visualSigns'][visual_sign]
    vs3 = json_data['labeller3']['visualSigns'][visual_sign]

    vs_vect = np.array([vs1, vs2, vs3])

    if gt_mode == 'median':
        vs = int(round(np.median(vs_vect) * max_value / label_max, 0))
    else:
        vs = int(round(np.mean(vs_vect) * max_value / label_max, 0))

    # Just in case there is an error
    if vs > max_value:
        vs = max_value
    return vs

def LegitDataloader(learn, labels_path, visual_signs, label_max, max_value, gt_mode):

    with open(labels_path) as json_file:
        json_data = json.load(json_file)

    # read all the filenames
    filenames = []
    for i in range(len(json_data)):
        filename = json_data['img' + str(i + 1)]['filename']
        filenames.append(filename)

    x_paths_train_new = []

    y_len = len(learn.data.train_ds.y)
    learn.data.train_ds.y = None
    multiple_label_list = []
    for i in range(y_len):
        x_path = learn.data.train_ds.x.items[i]
        filename = str(x_path).split('/')[-1]

        try:
            index = filenames.index(filename)

            vs_labels = []
            for i_sign in range(len(visual_signs)):
                vs = ground_truth_generator(json_data['img' + str(index + 1)],
                                            visual_signs[i_sign],
                                            max_value=max_value,
                                            label_max=label_max,
                                            gt_mode=gt_mode)
                vs_labels.append(vs)


            multiple_label_list.append([vs_labels[0],
                                        vs_labels[1],
                                        vs_labels[2],
                                        vs_labels[3],
                                        vs_labels[4],
                                        vs_labels[5]])
            x_paths_train_new.append(x_path)


        except:
            multiple_label_list.append([0, 0, 0, 0, 0, 0])
            x_paths_train_new.append(x_path)

    multiple_label_list = np.asarray(multiple_label_list)

    min_class = np.min(multiple_label_list)
    max_class = np.max(multiple_label_list)
    classes = [i for i in range(min_class, max_class + 1)]

    z = CategoryListAdapted(multiple_label_list, classes=classes)
    learn.data.train_ds.y = z

    # use only the images that are in the JSON
    learn.data.train_ds.x.items = None
    learn.data.train_ds.x.items = x_paths_train_new

    # Apply the same processing to the validation dataset
    try:
        x_paths_val_new = []

        y_len = len(learn.data.valid_ds.y)
        learn.data.valid_ds.y = None
        multiple_label_list = []
        for i in range(y_len):
            x_path = learn.data.valid_ds.x.items[i]
            filename = str(x_path).split('/')[-1]
            try:
                index = filenames.index(filename)

                vs_labels = []
                for i_sign in range(len(visual_signs)):
                    vs = ground_truth_generator(json_data['img' + str(index + 1)],
                                                visual_signs[i_sign],
                                                max_value=max_value,
                                                label_max=label_max,
                                                gt_mode=gt_mode)
                    vs_labels.append(vs)

                multiple_label_list.append([vs_labels[0],
                                            vs_labels[1],
                                            vs_labels[2],
                                            vs_labels[3],
                                            vs_labels[4],
                                            vs_labels[5]])
                x_paths_val_new.append(x_path)

            except:
                multiple_label_list.append([0, 0, 0, 0, 0, 0])
                x_paths_val_new.append(x_path)
    except:
        print('There is no validation data')


    multiple_label_list = np.asarray(multiple_label_list)
    z = CategoryListAdapted(multiple_label_list, classes=classes)
    learn.data.valid_ds.y = z

    # use only the images that are in the JSON
    learn.data.valid_ds.x.items = None
    learn.data.valid_ds.x.items = x_paths_val_new

    # Print traind and val image numbers
    print(len(learn.data.train_ds.x.items), len(learn.data.train_ds.y))
    print(len(learn.data.valid_ds.x.items), len(learn.data.valid_ds.y))
    return learn, len(classes)




