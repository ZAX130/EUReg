import random
import pickle
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchgeometry as tgm

M = 2 ** 32 - 1


def init_fn(worker):
    seed = torch.LongTensor(1).random_().item()
    seed = (seed + worker) % M
    np.random.seed(seed)
    random.seed(seed)


def add_mask(x, mask, dim=1):
    mask = mask.unsqueeze(dim)
    shape = list(x.shape);
    shape[dim] += 21
    new_x = x.new(*shape).zero_()
    new_x = new_x.scatter_(dim, mask, 1.0)
    s = [slice(None)] * len(shape)
    s[dim] = slice(21, None)
    new_x[s] = x
    return new_x


def sample(x, size):
    # https://gist.github.com/yoavram/4134617
    i = random.sample(range(x.shape[0]), size)
    return torch.tensor(x[i], dtype=torch.int16)
    # x = np.random.permutation(x)
    # return torch.tensor(x[:size])


def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


_shape = (240, 240, 155)


def get_all_coords(stride):
    return torch.tensor(
        np.stack([v.reshape(-1) for v in
                  np.meshgrid(
                      *[stride // 2 + np.arange(0, s, stride) for s in _shape],
                      indexing='ij')],
                 -1), dtype=torch.int16)


_zero = torch.tensor([0])


def gen_feats():
    x, y, z = 240, 240, 155
    feats = np.stack(
        np.meshgrid(
            np.arange(x), np.arange(y), np.arange(z),
            indexing='ij'), -1).astype('float32')
    shape = np.array([x, y, z])
    feats -= shape / 2.0
    feats /= shape

    return feats

def normalize(arr):
    return (arr - arr.min())/(arr.max() - arr.min())

def normalize_refmax(arr, ref_max):
    return (arr - arr.min())/(ref_max - arr.min())


class FrameRigidTransformer(nn.Module):
    def __init__(self, slice_size, mode='bilinear'):
        super(FrameRigidTransformer, self).__init__()
        '''
        slice_size:(h, w) (default vol size (T, H, W)) 
        mode: 'bilinear'(default), 'neareast'
        '''

        self.mode = mode
        self.slice_size = [1] + list(slice_size)
        vectors = [torch.linspace(-0.5 * (s - 1), 0.5 * (s - 1), steps=s) for s in self.slice_size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack([grids[2], grids[1], grids[0], torch.ones_like(grids[0])], dim=0) # G[x, y, z] = (z, y, x, 1)
        grid = grid.view(4, -1).type(torch.FloatTensor).contiguous()

        self.register_buffer('grid', grid)

    def dof2mat(self, input_dof):
        rad = tgm.deg2rad(input_dof[:, 3:])

        ai, aj, ak = rad[:, 0], rad[:, 1], rad[:, 2]
        si, sj, sk = torch.sin(ai), torch.sin(aj), torch.sin(ak)
        ci, cj, ck = torch.cos(ai), torch.cos(aj), torch.cos(ak)
        cc, cs = ci * ck, ci * sk
        sc, ss = si * ck, si * sk
        M = torch.eye(4, dtype=input_dof.dtype, device=input_dof.device).repeat(input_dof.shape[0], 1, 1)
        M[:, 0, 0] = cj * ck
        M[:, 0, 1] = sj * sc - cs
        M[:, 0, 2] = sj * cc + ss
        M[:, 1, 0] = cj * sk
        M[:, 1, 1] = sj * ss + cc
        M[:, 1, 2] = sj * cs - sc
        M[:, 2, 0] = -sj
        M[:, 2, 1] = cj * si
        M[:, 2, 2] = cj * ci
        M[:, :3, 3] = input_dof[:, :3]  # 平移分量

        return M

    def forward(self, vol, dof):  # DOF已经翻转的，比如volsize是H,W,D, 那么dof是[Td, Tw, Th, Rd, Rw, Rh]

        mat = self.dof2mat(dof)  # 已经是z, y, x顺序的
        new_locs = torch.matmul(mat, self.grid)[:, :3]  # (z, y, x, 1)->(z, y, x)

        vol_size = vol.shape[2:]
        for i in range(len(vol_size)): # to [-1, 1]
            new_locs[:, i] = 2 * ((new_locs[:, i] + 0.5 * (vol_size[2 - i] - 1)) / (vol_size[2 - i] - 1) - 0.5)

        new_locs = new_locs.permute(0, 2, 1).contiguous().view(vol.shape[0], *self.slice_size, 3)

        return F.grid_sample(vol, new_locs, align_corners=True, mode=self.mode)

def np2torch(arr):
    return torch.from_numpy(arr[np.newaxis, np.newaxis, ...]).float().cuda(0)

def torch2np(tensor):
    return tensor.squeeze().detach().cpu().numpy()