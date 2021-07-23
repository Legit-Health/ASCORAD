###########################################
# Author:     Alfonso Medela              #
# Contact     alfonso@legit.health        #
# Copyright:  Legit.Health                #
# Website:    https://legit.health/       #
###########################################

# Imports
from fastai.vision import *
import configparser

# Custom imports
sys.path.append('../../')
from utils.plot_utils import set_linewidth, plot_lesion_surface, plot_lesion_surface_annotation_masks
from utils.data_processing import generate_avg_masks

if __name__ == '__main__':

    # Load configuration file
    root = '../../'
    SERVABLE_CFG_FILE = root + 'config.ini'
    config = configparser.ConfigParser()
    config.read(SERVABLE_CFG_FILE)

    # Get paths
    figure_output_path = config['DATA']['FIGURES_ROOT_PATH']

    # Choose output folder and create it if it doesn't exist
    output_folder = figure_output_path + 'exp1/'

    # Set the path to the image
    image_path = 'IMAGE_PATH'
    mask_path = 'MASKS_PATH'

    # Load image
    PIL_img = PIL.Image.open(image_path).convert('RGB')

    # Annotations
    masks, mask_mean = generate_avg_masks(image_path, mask_path)

    # Individial Annotations
    for i_mask in range(len(masks)):
        output_img, _, _ = plot_lesion_surface_annotation_masks(PIL_img, masks[i_mask])
        output_img = PIL.Image.fromarray(output_img)
        output_img.save(output_folder + 'mask_' + str(i_mask) + '.png', "PNG")

    # Mean mask = GT
    output_img, _, _ = plot_lesion_surface_annotation_masks(PIL_img, mask_mean)
    output_img = PIL.Image.fromarray(output_img)
    output_img.save(output_folder + 'mask_GT.png', "PNG")

    # Predict
    learner_path = 'LEARNER_PATH'
    ASCORAD_LS_learner = load_learner(learner_path)

    output_img, lesion_image, mask = plot_lesion_surface(PIL_img, ASCORAD_LS_learner)
    output_img = PIL.Image.fromarray(output_img)
    output_img.save(output_folder + 'ASCORAD_pred.png', "PNG")

