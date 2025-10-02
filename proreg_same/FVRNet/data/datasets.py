import os, glob

import torch, sys
from torch.utils.data import Dataset
from .data_utils import pkload, normalize, normalize_refmax, FrameRigidTransformer, np2torch
import random
import numpy as np
import re
import numpy as np

import torch, sys
from torch.utils.data import Dataset
from torch import nn

class TrainDataset(Dataset):
    def __init__(self, data_path,vol_size, slice_size, scaler=[10,10,10,10,10,10], delta=0.75, lower_bound=0.11):
        self.paths = data_path
        self.vol_size = vol_size
        self.scaler = torch.tensor([scaler],dtype=torch.float).cuda()    # _,_,frame,_,_,frame
        self.frt = FrameRigidTransformer(slice_size).cuda()
        self.delta = delta
        self.lower_bound = lower_bound


    def __getitem__(self, index):
        path = self.paths[index]
        vol, _ = pkload(path)

        ori_shape = vol.shape
        vol_tensor = np2torch(vol)
        while True: # random sample
            dof = torch.rand(1, 6, device=vol_tensor.device)
            dof = 2 * (dof - 0.5) * self.scaler

            rslice_tensor = self.frt(vol_tensor, dof)
            rslice_score = (rslice_tensor > self.lower_bound).sum() / np.prod(rslice_tensor.shape)

            if rslice_score > self.delta:
                return vol_tensor[0].contiguous(), rslice_tensor[0].contiguous(), dof[0].contiguous()

    def __len__(self):
        return len(self.paths)

class TrainDataset_postcut(Dataset):
    def __init__(self, data_path,vol_size, slice_size, scaler=[10,10,10,10,10,10], delta=0.75, lower_bound=0.11):
        self.paths = data_path
        self.vol_size = vol_size
        self.scaler = torch.tensor([scaler],dtype=torch.float).cuda()    # _,_,frame,_,_,frame
        self.frt = FrameRigidTransformer(slice_size).cuda()
        self.delta = delta
        self.lower_bound = lower_bound


    def __getitem__(self, index):
        path = self.paths[index]
        vol, _ = pkload(path)

        ori_shape = vol.shape
        vol_tensor = np2torch(vol)
        while True: # random sample
            dof = torch.rand(1, 6, device=vol_tensor.device)
            dof = 2 * (dof - 0.5) * self.scaler

            rslice_tensor = self.frt(vol_tensor, dof)
            rslice_score = (rslice_tensor > self.lower_bound).sum() / np.prod(rslice_tensor.shape)

            if rslice_score > self.delta:
                return vol_tensor[0,:,:,(80-64)//2:(80+64)//2, (96-64)//2:(96+64)//2].contiguous(), rslice_tensor[0].contiguous(), dof[0].contiguous()

    def __len__(self):
        return len(self.paths)

class TestDataset(Dataset):
    def __init__(self, data_path):
        self.paths = data_path

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i, ...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        vol, mask, slice, slice_mask, dof = pkload(path)

        vol = np.ascontiguousarray(vol[None, ...])

        slice = np.ascontiguousarray(slice[None,None,...])
        param = np.ascontiguousarray(dof)

        vol, slice, param = torch.from_numpy(vol), torch.from_numpy(slice), torch.from_numpy(param)
        return vol, slice, param
    def __len__(self):
        return len(self.paths)

