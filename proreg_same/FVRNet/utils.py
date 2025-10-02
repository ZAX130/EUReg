import logging

import numpy as np
from torch import Tensor
import torch.nn.functional as f
import torch
import torchgeometry as tgm

def dof6mat_tensor(input_dof, device):

    rad = tgm.deg2rad(input_dof[:, 3:])

    ai = rad[:, 0]
    aj = rad[:, 1]
    ak = rad[:, 2]

    si, sj, sk = torch.sin(ai), torch.sin(aj), torch.sin(ak)
    ci, cj, ck = torch.cos(ai), torch.cos(aj), torch.cos(ak)
    cc, cs = ci*ck, ci*sk
    sc, ss = si*ck, si*sk

    M = torch.zeros((input_dof.shape[0], 4, 4)).cuda()

    M[:, 0, 0] = cj*ck
    M[:, 0, 1] = sj*sc-cs
    M[:, 0, 2] = sj*cc+ss
    M[:, 1, 0] = cj*sk
    M[:, 1, 1] = sj*ss+cc
    M[:, 1, 2] = sj*cs-sc
    M[:, 2, 0] = -sj
    M[:, 2, 1] = cj*si
    M[:, 2, 2] = cj*ci
    M[:, :3, 3] = input_dof[:, :3]
    M[:, 3, 3] = 1

    return M


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = []
        self.std = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.vals.append(val)
        self.std = np.std(self.vals)

def dice_val_VOI(y_pred, y_true):
    VOI_lbls = [1, 2, 3]
    pred = y_pred.detach().cpu().numpy()[0, 0, ...]
    true = y_true.detach().cpu().numpy()[0, 0, ...]
    DSCs = np.zeros((len(VOI_lbls), 1))
    idx = 0
    for i in VOI_lbls:
        pred_i = pred == i
        true_i = true == i
        intersection = pred_i * true_i
        intersection = np.sum(intersection)
        union = np.sum(pred_i) + np.sum(true_i)
        dsc = (2.*intersection) / (union + 1e-5)
        DSCs[idx] =dsc
        idx += 1
    return np.mean(DSCs)