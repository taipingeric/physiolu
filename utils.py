import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.metrics import auc, roc_curve, confusion_matrix, classification_report, f1_score

def evaluation(model, dataloader, device):
    model.eval()
    y_pred = np.empty((0, 3))
    y_true = np.array([])
    
    with torch.no_grad():
        for batch_i, (x, y) in enumerate(tqdm(dataloader, leave=False)):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            pred = pred.cpu().numpy()
            y = y.cpu().numpy()
            
            y_pred = np.concatenate((y_pred, pred))
            y_true = np.concatenate((y_true, y))
            
    
    y_pred = np.argmax(np.array(y_pred), axis=-1)
    print(y_true.shape, y_pred.shape)
    print(f1_score(y_true, y_pred, average='macro'))
    print(classification_report(y_true, y_pred, digits=3))


def center_crop(tensor):
    length = len(tensor)
    offset = length // 4
    return tensor[offset: -offset]

# PyTorch Dice loss
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, padding=False):
        super(DiceLoss, self).__init__()
        self.padding = padding

    def forward(self, inputs, targets, smooth=1):
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        if self.padding:
            inputs = center_crop(inputs)
            targets = center_crop(targets)
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        return 1 - dice
    
# PyTorch IoU
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True, loss=True, padding=False):
        super(IoULoss, self).__init__()
        self.padding = padding
        self.loss = loss

    def forward(self, inputs, targets, smooth=1):
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        if self.padding:
            inputs = center_crop(inputs)
            targets = center_crop(targets)
        # intersection: True Positive count
        # union: mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        iou = (intersection + smooth)/(union + smooth)
        # return IoU loss or IoU value
        if self.loss:
            return 1 - iou
        else:
            return iou
        
def convert_py_str(file_path):
    with open (file_path, "r", encoding="utf-8") as f:
        data = ' '.join([line for line in f.readlines()])
    return data