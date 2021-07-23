###########################################
# Author:     Alfonso Medela              #
# Contact     alfonso@legit.health        #
# Copyright:  Legit.Health                #
# Website:    https://legit.health/       #
###########################################

# Imports
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import configparser

def main(output_folder, json_data, visual_signs, number_of_annotators, color = (9 / 255, 169 / 255, 95 / 255)):

    print('Running...')
    vs_results_dataset = []
    img_num = len(json_data)
    for i_img in range(1, img_num + 1):

        # Visual signs
        vs_results = np.empty([len(visual_signs), number_of_annotators])
        for i_vs in range(len(visual_signs)):
            for i_ann in range(1, number_of_annotators + 1):
                vs_n = json_data['img' + str(i_img)]['labeller' + str(i_ann)]['visualSigns'][visual_signs[i_vs]]
                vs_results[i_vs, i_ann - 1] = vs_n

        vs_results_dataset.append(vs_results)

    vs_results_dataset = np.asarray(vs_results_dataset)

    GT_mean = np.mean(vs_results_dataset, axis=-1)
    GT_median = np.median(vs_results_dataset, axis=-1)

    for i_vs in range(len(visual_signs)):
        GT_mean_vs = np.unique(GT_mean[:, i_vs], return_counts=True)

        x = GT_mean_vs[0]
        bins = GT_mean_vs[1]

        plt.bar(x, bins, 0.4, color=color)
        plt.title(visual_signs[i_vs] + ' intensity distribution')
        plt.savefig(output_folder + 'mean/' + visual_signs[i_vs] + '_mean.png')
        plt.gcf().clear()

        GT_median_vs = np.unique(GT_median[:, i_vs], return_counts=True)

        x = GT_median_vs[0]
        bins = GT_median_vs[1]

        plt.bar(x, bins, 0.4, color=color)
        plt.title(visual_signs[i_vs] + ' intensity distribution')
        plt.savefig(output_folder + 'median/' + visual_signs[i_vs] + '_median.png')
        plt.gcf().clear()


if __name__ == '__main__':

    # Load configuration file
    root = '../../'
    SERVABLE_CFG_FILE = root + 'config.ini'
    config = configparser.ConfigParser()
    config.read(SERVABLE_CFG_FILE)

    # Input params defined in config file
    number_of_annotators = int(config['ANNOTATION']['NUMBER_OF_ANNOTATORS'])

    # Load visual signs
    vs_num = int(config['ANNOTATION']['VISUAL_SIGN_NUMBER'])
    visual_signs = []
    for i_vs in range(vs_num):
        vs = config['ANNOTATION']['VISUAL_SIGN_' + str(i_vs) + '']
        visual_signs.append(vs)

    root_path = config['DATA']['DATASET_ROOT_PATH']
    path = root_path + 'LegitHealth-AD/'

    # open json
    json_path = path + 'labels/visual_sign_assessment/visual_sign_intensities.json'
    with open(json_path) as json_file:
        json_data = json.load(json_file)

    # Load output path
    output_path = root + '/data/results/distributions/plots/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    main(output_path, json_data, visual_signs, number_of_annotators)






