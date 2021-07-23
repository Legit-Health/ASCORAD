###########################################
# Author:     Alfonso Medela              #
# Contact     alfonso@legit.health        #
# Copyright:  Legit.Health                #
# Website:    https://legit.health/       #
###########################################

from fastai.vision import *
from efficientnet_pytorch import EfficientNet

class Network(nn.Module):
    def __init__(self, num_classes, device):
        super().__init__()

        model_name = 'efficientnet-b0'
        self.model = EfficientNet.from_pretrained(model_name)

        self.classes = num_classes


        # modify the model
        self.model._fc = nn.Identity()
        self.model._swish = nn.Identity()

        self.device = device
        self.model.to(self.device)

        # Six head for six visual signs
        self.l1 = nn.Linear(1280, self.classes).to(self.device)
        self.l2 = nn.Linear(1280, self.classes).to(self.device)
        self.l3 = nn.Linear(1280, self.classes).to(self.device)
        self.l4 = nn.Linear(1280, self.classes).to(self.device)
        self.l5 = nn.Linear(1280, self.classes).to(self.device)
        self.l6 = nn.Linear(1280, self.classes).to(self.device)

    def forward(self, x):

        # Run the shared layer(s)
        x = self.model(x)

        # Run the different heads with the output of the shared layers as input
        out_1 = self.l1(x)
        out_2 = self.l2(x)
        out_3 = self.l3(x)
        out_4 = self.l4(x)
        out_5 = self.l5(x)
        out_6 = self.l6(x)

        return out_1, out_2, out_3, out_4, out_5, out_6

class MultipleHeadCrossEntropy(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, input, target, target2, target3, target4, target5, target6, **kwargs):

        ce1 = torch.nn.functional.cross_entropy(input[0], target)
        ce2 = torch.nn.functional.cross_entropy(input[1], target2)
        ce3 = torch.nn.functional.cross_entropy(input[2], target3)
        ce4 = torch.nn.functional.cross_entropy(input[3], target4)
        ce5 = torch.nn.functional.cross_entropy(input[4], target5)
        ce6 = torch.nn.functional.cross_entropy(input[5], target6)

        multiple_head_crossentropy = (ce1 + ce2 + ce3 + ce4 + ce5 + ce6) / 6.
        return multiple_head_crossentropy

class CategoryListBase(ItemList):
    "Basic `ItemList` for classification."
    def __init__(self, items:Iterator, classes:Collection=None, **kwargs):
        self.classes=classes
        self.filter_missing_y = True
        super().__init__(items, **kwargs)
        self.copy_new.append('classes')

    @property
    def c(self): return len(self.classes)

class CategoryListAdapted(CategoryListBase):
    "Basic `ItemList` for single classification labels."
    _processor=CategoryProcessor
    def __init__(self, items:Iterator, classes:Collection=None, label_delim:str=None, **kwargs):
        super().__init__(items, classes=classes, **kwargs)
        self.loss_func = CrossEntropyFlat()
        self.classes = classes

    def get(self, i):
        o = self.items[i]
        if o is None: return None
        return [Category(o[0], self.classes[o[0]]),
                Category(o[1], self.classes[o[1]]),
                Category(o[2], self.classes[o[2]]),
                Category(o[3], self.classes[o[3]]),
                Category(o[4], self.classes[o[4]]),
                Category(o[5], self.classes[o[5]])]

    def analyze_pred(self, pred, thresh:float=0.5): return pred

    def reconstruct(self, t):
        return 0

class FMAPE(Callback):
    # Needs to run before the recorder
    _order = -20
    def __init__(self, learn, **kwargs):
        self.learn = learn

    def on_train_begin(self, **kwargs):
        self.learn.recorder.add_metric_names(['RMAE',
                                              'RMAE-red',
                                              'RMAE-cru',
                                              'RMAE-swe',
                                              'RMAE-scr',
                                              'RMAE-lic',
                                              'RMAE-dry'])

    def on_epoch_begin(self, **kwargs):

        # Final MAE
        self.mae = 0

        # MAE by visual sign (6)
        self.mae_total = [0]*6
        self.times_summed_total = [0]*6

    def on_batch_end(self, last_target, last_output, train, **kwargs):

        self.output = last_output
        self.target = last_target

        for i_sign in range(6):
            # Visual sign stats
            preds = self.output[i_sign].argmax(1)
            target = self.target[i_sign]
            for j in range(len(preds)):
                self.mae_total[i_sign] = self.mae_total[i_sign] + torch.abs(preds[j] - target[j]) * 10
                self.times_summed_total[i_sign] += 1

    def on_epoch_end(self, last_metrics, **kwargs):

        for i_sign in range(len(self.mae_total)):
            mae = self.mae_total[i_sign].cpu().detach().numpy() / self.times_summed_total[i_sign]
            self.mae += mae

        self.mae = self.mae / len(self.mae_total)

        out_metrics = [self.mae]
        for i_sign in range(len(self.mae_total)):
            mae = self.mae_total[i_sign].cpu().detach().numpy() / self.times_summed_total[i_sign]
            out_metrics.append(mae)

        return add_metrics(last_metrics, out_metrics)

def softmax(x):
    '''
    Compute softmax values for each sets of scores in x
    :param x: input vector
    :return: output vector
    '''

    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
