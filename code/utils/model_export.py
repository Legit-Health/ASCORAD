###########################################
# Author:     Alfonso Medela              #
# Contact     alfonso@legit.health        #
# Copyright:  Legit.Health                #
# Website:    https://legit.health/       #
###########################################

# Imports
from fastai.vision import *

def custom_export(custom_path, learn):

    '''
    Custom export function
    :param custom_path: Path to save the model
    :param learn: learner
    :return: None
    '''

    args = ['opt_func', 'loss_func', 'metrics', 'true_wd', 'bn_wd', 'wd', 'train_bn', 'model_dir', 'callback_fns']
    state = {a: getattr(learn, a) for a in args}
    state['cb_state'] = {cb.__class__: cb.get_state() for cb in learn.callbacks}
    with ModelOnCPU(learn.model) as m:
        state['model'] = m
        xtra = dict(normalize=learn.data.norm.keywords) if getattr(learn.data, 'norm', False) else {}
        state['data'] = learn.data.valid_ds.get_state(**xtra)
        state['cls'] = learn.__class__
        custom_path = pathlib.PosixPath(custom_path)
        try_save(state, custom_path, 'export.pkl')