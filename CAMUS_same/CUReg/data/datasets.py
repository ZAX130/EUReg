import os, glob
import torch, sys
from torch.utils.data import Dataset
from .data_utils import pkload, normalize
import random
import numpy as np
import re

class CURegTrainDataset(Dataset):
    def __init__(self, data_path):
        self.paths = data_path

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out
    def get_id(self, path):
        vol_id, slice_id = re.findall(r"\d+", os.path.basename(path))
        return int(vol_id), int(slice_id)
    def __getitem__(self, index):
        path = self.paths[index]
        vol, slice, seg, param, dist = pkload(path)
        # print(path, end=' ')
        vol_id, slice_id = self.get_id(path)
        slices = [slice]
        for id in range(4):
            if id == slice_id :
                continue
            slice_path = os.path.join(os.path.dirname(path), 'vol_{}_slice_{}.pkl'.format(vol_id, id))
            # print(slice_path , end=' ')
            slices.append(pkload(slice_path)[1])
        slices = np.stack(slices, axis=0)
        # print(' ')
        vol = vol[None, ...]
        slices = slices[:,None,...]
        seg = seg[None, ...]

        vol = np.ascontiguousarray(vol)
        slices = np.ascontiguousarray(slices)
        seg = np.ascontiguousarray(seg)
        param = np.ascontiguousarray(param)
        dist = np.ascontiguousarray(dist)

        vol, slices, seg, param, dist = torch.from_numpy(vol), torch.from_numpy(slices), torch.from_numpy(seg), torch.from_numpy(param), torch.from_numpy(dist)
        return vol, slices, seg, param, dist

    def __len__(self):
        return len(self.paths)


class CURegInferDataset(Dataset):
    def __init__(self, data_path):
        self.paths = data_path

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i, ...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        vol, slice, seg, param, dist = pkload(path)
        vol = vol[None, ...]
        slice = slice[None, None, ...]
        # vol = normalize(vol[None, ...])
        # slice = normalize(slice[None, None, ...])
        seg = seg[None, None, ...]

        vol = np.ascontiguousarray(vol)
        slice = np.ascontiguousarray(slice)
        seg = np.ascontiguousarray(seg)
        param = np.ascontiguousarray(param)
        dist = np.ascontiguousarray(dist)

        vol, slice, seg, param, dist = torch.from_numpy(vol), torch.from_numpy(slice), torch.from_numpy(
            seg), torch.from_numpy(param), torch.from_numpy(dist)
        return vol, slice, seg, param, dist

    def __len__(self):
        return len(self.paths)

class CURegOriTrainDataset(Dataset):
    def __init__(self, data_path):
        self.paths = data_path
        self.all_vol_indexs = np.arange(len(data_path)//4)

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out
    def get_sliceid(self):
        all_slices = [0, 1, 2, 3]
        random.shuffle(all_slices)
        return all_slices

    def __getitem__(self, index):
        vol_id = self.all_vol_indexs[index]
        slice_ids = self.get_sliceid()
        path = os.path.join(os.path.dirname(self.paths[index]), 'vol_{}_slice_{}.pkl'.format(vol_id, slice_ids[0]))

        vol, slice, seg, param, dist = pkload(path)


        slices = [normalize(slice)]
        for id in sorted(slice_ids[1:]):
            slice_path = os.path.join(os.path.dirname(path), 'vol_{}_slice_{}.pkl'.format(vol_id, id))
            slices.append(normalize(pkload(slice_path)[1]))
        slices = np.stack(slices, axis=0)

        vol = normalize(vol[None, ...])
        slices = slices[:,None,...]
        seg = seg[None, ...]

        vol = np.ascontiguousarray(vol)
        slices = np.ascontiguousarray(slices)
        seg = np.ascontiguousarray(seg)
        param = np.ascontiguousarray(param)
        dist = np.ascontiguousarray(dist)

        vol, slices, seg, param, dist = torch.from_numpy(vol), torch.from_numpy(slices), torch.from_numpy(seg), torch.from_numpy(param), torch.from_numpy(dist)
        return vol, slices, seg, param, dist

    def __len__(self):
        return len(self.all_vol_indexs)