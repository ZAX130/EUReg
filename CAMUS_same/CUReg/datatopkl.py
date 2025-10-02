import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import random
import time
import numpy as np

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from datasets.s2v_dataset import MyDataset as PoseDataset_s2v
from networks.dual_fusionNet import RegistNetwork
from lib.loss import regularization_loss
from lib.utils import setup_logger
from tensorboardX import SummaryWriter
from batchgenerators.utilities.file_and_folder_operations import makedirs

dataset_root = '/data/whq/Data/CAMUS'
test_dataset = PoseDataset_s2v('test', dataset_root)
# testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=10)
import pickle
# def pksave(data, save_path):
#     with open(save_path, 'wb') as f:
#         pickle.dump(data, f)
# for i, data in enumerate(test_dataset):
#     print(data[0])
#     # print(data[1].shape)  # vol
#     # print(data[2].shape)  # slice
#     # print(data[3].shape)  # mat
#     # print(data[4].shape, data[4])  # 6 dof
#     # print(data[5].shape, data[5])  # 3 tradist
#     # print(data[6].shape)  #
#
#     all_data = []
#     all_data.append(data[1].squeeze().numpy())
#     all_data.append(data[2].squeeze().numpy())
#     all_data.append(data[6].squeeze().numpy())
#     all_data.append(data[4].squeeze().numpy())
#     all_data.append(data[5].squeeze().numpy())
#     # vol slice mask 6dof 3dist
#     for d in all_data:
#         print(d.shape)
#     save_path = '/data/whq/Data/CAMUS_data/Test/vol_{}_slice_{}.pkl'.format(*data[0])
#
#     pksave(all_data, save_path)

import numpy as np

A = np.array([[1,0,1,0], [2,3,3,4]])

print(A[None,...].repeat(4, axis=0))
