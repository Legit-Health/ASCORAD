# ASCORAD: Automatic Scoring of Atopic Dermatitis
[![fastai](https://img.shields.io/badge/fastai-1.0.61-blue?style=plastic)](https://www.fast.ai/)
[![torch](https://img.shields.io/badge/torch-1.6.0-orange?style=plastic)](https://pytorch.org/)
[![opencv](https://img.shields.io/badge/opencv--python-4.4.0.44-brightgreen?style=plastic)](https://opencv.org/)

[<img src="figures/Legit_Health_logo.png" width="500" height="70" />](https://legit.health/)

## Table of contents
- [What is ASCORAD?](#what-is-ascorad)
- [Datasets](#datasets)
- [Method](#method)
- [Results](#results)
- [The paper](#reference)
- [Contact](#contact)


## [What is ASCORAD?](https://legit.health/)
ASCORAD is a fast, accurate and fully automatic scoring system for the severity assessment of atopic dermatitis.

![Legit Health Web Application](figures/Figure_3.png)

## Datasets

*SCORADNet* was train and tested using three datasets:

- **LegitHealth-AD**. 604 images that belong to Caucasian patients, of which one third are children, suffering from atopic
dermatitis with lesions present on different body parts.
- **LegitHealth-AD-Test**. 367 images that belong to Caucasian patients and were gathered from several dermatological atlases available online. 
- **LegitHealth-AD-FPK-IVI**. 112 images collected from online dermatological atlases that contains photos of patients with IV, V and VI skin types suffering from atopic dermatitis.


In order to run the code, the datasets have to be placed in a directory (*DATASET_ROOT_PATH*) following this scheme:

```
├── 📁 DATASET_ROOT_PATH
│   |
│   ├── 📁 LegitHealth-AD
│   │   ├── 📁 images
│   │   ├── 📁 labels
│   |   |   ├── 📁 visual_sign_assessment
│   |   |   ├── 📁 lesion_segmentation
│   |   |   |   ├── 📁 ground_truth_masks
│   |   |   |   ├── 📁 masks
|   |
│   ├── 📁 LegitHealth-AD-Test
│   ├── 📁 LegitHealth-AD-FPK-IVI
│   ├── 📁 LegitHealth-HealthySkin
```

The distributions of the visual sign intensities are quite different on each dataset, as it can be seen in the following figure:

![Dataset distributions](figures/Figure_1.png)


## Method
The ASCORAD calculation can be divided in two parts, lesion surface segmentation and visual sign severity assessment. We trained two separated models, one for each task, and named SCORADNet to the neural networks involved in the calculation of the ASCORAD. We used a **U-Net** with **Resnet-34** backbone for segmentation and a multi-output (6) **EfficientNet-B0** for classification.

## Results
### Lesion segmentation

Resultson on Caucasian skin (LegitHealth-AD-Test) LegitHealth-AD-Test are very promising, obtaining an IoU greater than 50%, 75% F1 and an outstanding AUC of 93%.

Metrics on LegitHealth-AD-Test and LegitHealth-AD-FPK-IVI
|  Skin type      | Accuracy | AUC   | IoU   |   F1  |
|:--------------:|:--------:|-------|-------|:-----:|
| Fitzpatric I,II,III |   84.54  | 93.06 | 63.86 | 74.45 |
| Fitzpatric IV,V,VI |   79.18  | 86.57 | 44.86 | 55.21 |

![Dataset distributions](figures/Figure_4.png)


[Results](https://github.com/Legit-Health/ASCORAD/blob/main/code/lesion-segmentation/exp2/readme.md) on darker skin improve including dark skin samples on the training set, however, metrics do not get as good as with Causasian skin. This is due to the small amount of images of dark skin. We have proved that results improve adding just a few images, and this is promising to create a model that performs well on the whole Fitzpatrick scale in the future.


There is a significant difference between the model from experiment 1 to the second one on darker skin, which can be seen in the following example. The image from the left shows the predicted mask from experiment 1 and the second one the mask predicted with the model trained on experiment 2. In the second case, the image corresponds to the test split.

![Dataset distributions](figures/Figure_5.png)

### Visual sign severity grading
#### Finding the optimal parameters
In [experiment 1](https://github.com/Legit-Health/ASCORAD/tree/main/code/visual-sign-intensity-estimation/exp1) we proved the hypothesis that a larger range contributes to lower RMAE values. We obtained 0.7-1% better results with [0, 100] range that in the original [0, 3] range. In the [second experiment](https://github.com/Legit-Health/ASCORAD/tree/main/code/visual-sign-intensity-estimation/exp2) we found out that training with the median of the annotators as the ground truth gets better results. Finally, we also demonstrated that despite obtaining a similar RMAE, [upsampling](https://github.com/Legit-Health/ASCORAD/tree/main/code/visual-sign-intensity-estimation/exp3) the dataset leads to significantly better predicted intensity distributions, predicting values from both extremes of the distribution.

| Experiment |  Range  | GT statistic | Upsampling | RMAE 1 (v2) | RMAE 2 (v2) | RMAE 1 (v3) | RMAE 2 (v3) |
|------------|:-------:|:------------:|:----------:|:-----------:|:-----------:|:-----------:|:-----------:|
| 1          |  [0,3]  |    Median    |     No     |     13.2    |     13.9    |     19.6    |     18.5    |
| 1          |  [0,10] |    Median    |     No     |     14.7    |     12.5    |     21.1    |     18.0    |
| 1          | [0,100] |    Median    |     No     |     14.5    |     12.2    |     21.0    |     17.8    |
| 2          | [0,100] |     Mean     |     No     |     13.8    |     12.6    |     21.1    |     17.8    |


Usin the range [0, 100], the median GT only for training, and no upsampling gave the best RMAE result, with 12.2% for LegitHealth-AD-Test and 17.8% for LegitHealth-AD-FPK-IVI.


## Reference

## Contact
[Alfonso Medela](https://www.linkedin.com/notifications/) \
[Taig Mac Carthy](https://www.linkedin.com/in/taigmaccarthy/) \
[Andy Aguilar](https://www.linkedin.com/in/andy-aguilar/) 
