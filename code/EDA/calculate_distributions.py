###########################################
# Author:     Alfonso Medela              #
# Contact     alfonso@legit.health        #
# Copyright:  Legit.Health                #
# Website:    https://legit.health/       #
###########################################

# Imports
import json
import numpy as np
import pandas as pd
import configparser
from tqdm import tqdm
import os

def main(output_folder, json_data, visual_signs, number_of_annotators):

    '''
    Main fuction for visual sign intensity distribution calculation
    :param output_folder: Results are saved in this directory
    :param json_data: Intensity annotations
    :param visual_signs: List of visual signs like erythema (redness)
    :param number_of_annotators: The number of labellers that annotated the dataset
    :return: None
    '''

    vs_results_dataset = []
    img_num = len(json_data)
    for i_img in tqdm(range(1, img_num + 1)):
        # Visual signs
        vs_results = np.empty([len(visual_signs), number_of_annotators])
        for i_vs in range(len(visual_signs)):
            for i_ann in range(1, number_of_annotators + 1):
                vs_n = json_data['img' + str(i_img)]['labeller' + str(i_ann)]['visualSigns'][visual_signs[i_vs]]
                vs_results[i_vs, i_ann - 1] = vs_n

        vs_results_dataset.append(vs_results)

    vs_results_dataset = np.asarray(vs_results_dataset)

    # Get the ground truth (mean)
    GT_mean = np.mean(vs_results_dataset, axis=-1)

    for i_vs in range(len(visual_signs)):
        # Get unique values and the counts for each one
        GT_mean_vs = np.unique(GT_mean[:, i_vs], return_counts=True)
        GT_mean_vs = np.asarray(GT_mean_vs)

        # Normalize the distribution [0, 1]
        GT_mean_vs[-1] = GT_mean_vs[-1] / np.max(GT_mean_vs[-1])

        # Convert it to a dataframe and save it as csv
        GT_mean_vs = pd.DataFrame(GT_mean_vs)
        GT_mean_vs.to_csv(output_folder + visual_signs[i_vs] + '_mean.csv')


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
            json_data = json.load(json_file)

        # Create folder if it doesn't exist
        output_folder = root + results_path + 'distributions/dataframes/' + dataset_name + '/'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Run main function for each dataset
        main(output_folder, json_data, visual_signs, number_of_annotators)






