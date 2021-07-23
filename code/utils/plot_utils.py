###########################################
# Author:     Alfonso Medela              #
# Contact     alfonso@legit.health        #
# Copyright:  Legit.Health                #
# Website:    https://legit.health/       #
###########################################

# imports
from fastai.vision import *
import imutils
import cv2

def set_linewidth(orig_img, per=0.005):

    '''
    Function to set a width for the plotting function
    :param orig_img: Input image
    :param per: percentage
    :return: line width
    '''

    a, b = orig_img.shape[0], orig_img.shape[1]
    line_width = int(round(b*per, 0))
    return line_width

def plot_lesion_surface(image, learner, a1=0.95, a2=0.95,  t=100):

    '''
    Function to plot the predicted lesion surface
    :param image: Input image
    :param learner: Segmentation learner
    :param a1: predefined parameters, use default
    :param a2: predefined parameters, use default
    :param t: threshold
    :return: image with the mask drawn, image with the lesion segmented and background set to black, the resultant mask
    '''

    # From array to PIL image. First we reshape to a perfect square.
    orig_img = np.asarray(image)
    a = orig_img.shape[0]
    b = orig_img.shape[1]
    if a > b:
        img = cv2.resize(orig_img, (b, b))
    else:
        img = cv2.resize(orig_img, (a, a))
    img = PIL.Image.fromarray(img).convert('RGB')
    img = pil2tensor(img, np.float32)
    img = img.div_(255)
    img = Image(img)

    _, _, mask = learner.predict(img)

    # Mask processing
    orig_img = np.asarray(image)
    mask = mask.detach().numpy()
    mask = mask[1]
    mask = cv2.resize(mask, (orig_img.shape[1], orig_img.shape[0]))
    mask = mask * 255
    mask = mask.astype(np.uint8)
    mask_output = mask.copy()
    mask = mask[:, :, np.newaxis]

    # Overlay mask and original image
    black = np.zeros([mask.shape[0], mask.shape[1], 1]).astype(np.uint8)
    mask = np.concatenate((black, black, mask), axis=-1)
    dst = cv2.addWeighted(orig_img, a1, mask, a2, 0)

    # Binarize the mask with a certain threshold
    mask[mask < t] = 0
    mask[mask >= t] = 255
    mask = mask[:, :, -1]

    # Plot parameters: line color and width
    line_color = (134, 113, 255)
    line_width = set_linewidth(orig_img)

    # Find all the contours in the mask and draw the boundary line with the color and width defined before
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    for c in cnts:
        cv2.drawContours(dst, [c], -1, line_color, line_width)

    output_mask = mask_output[:, :, np.newaxis]
    output_mask = np.concatenate((output_mask, output_mask, output_mask), axis=-1)
    output_mask = output_mask / 255
    output_mask[output_mask <= 0.5] = 0
    output_mask[output_mask > 0.5] = 1
    output_mask = output_mask.astype(int)

    lesion_image = orig_img * output_mask
    return dst, lesion_image, mask

def plot_lesion_surface_annotation_masks(image, mask, a1=0.95, a2=0.95,  t=100):

    '''
    Function to plot the mask on the image
    :param image: Input image
    :param mask: Input mask
    :param a1: predefined parameters, use default
    :param a2: predefined parameters, use default
    :param t: threshold
    :return: image with the mask drawn, image with the lesion segmented and background set to black, the resultant mask
    '''

    # Mask processing
    orig_img = np.asarray(image)
    mask = cv2.resize(mask, (orig_img.shape[1], orig_img.shape[0]))
    mask = mask * 255
    mask = mask.astype(np.uint8)
    mask_output = mask.copy()
    mask = mask[:, :, np.newaxis]

    # Overlay mask and original image
    black = np.zeros([mask.shape[0], mask.shape[1], 1]).astype(np.uint8)
    mask = np.concatenate((black, black, mask), axis=-1)
    dst = cv2.addWeighted(orig_img, a1, mask, a2, 0)

    # Binarize the mask with a certain threshold
    mask[mask < t] = 0
    mask[mask >= t] = 255
    mask = mask[:, :, -1]

    # Plot parameters: line color and width
    line_color = (134, 113, 255)
    line_width = set_linewidth(orig_img)

    # Find all the contours in the mask and draw the boundary line with the color and width defined before
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    for c in cnts:
        cv2.drawContours(dst, [c], -1, line_color, line_width)

    output_mask = mask_output[:, :, np.newaxis]
    output_mask = np.concatenate((output_mask, output_mask, output_mask), axis=-1)
    output_mask = output_mask / 255
    output_mask[output_mask <= 0.5] = 0
    output_mask[output_mask > 0.5] = 1
    output_mask = output_mask.astype(int)

    lesion_image = orig_img * output_mask
    return dst, lesion_image, mask
