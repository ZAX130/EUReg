import glob
import losses
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random
import sys
import utils

import numpy as np
import torch
import torch.nn as nn
from natsort import natsorted
from torch import optim
from torch.utils.data import DataLoader
from pytorch_msssim import SSIM
from data.datasets import CURegInferDataset
from networks.dual_fusionNet import RegistNetwork


def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


same_seeds(2032)


class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir + "logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


GPU_iden = 0


def main():
    val_dir = 'Data/CAMUS_data/Test/'

    dim = 12
    save_dir = 'CUReg/'
    img_size = (32, 128, 128)

    '''
    Initialize model
    '''
    model = RegistNetwork(layers=[3, 8, 36, 3])
    model.cuda()

    model_dir = 'experiments/miccai2025/CAMUS_same/' + save_dir
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[0], map_location='cuda:0')
    model.load_state_dict(best_model)
    print(model_dir + natsorted(os.listdir(model_dir))[0])
    del best_model

    '''
    DataLoader
    '''
    test_set = CURegInferDataset(glob.glob(val_dir + '*.pkl'))
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=10, pin_memory=True)

    '''
    Criterion
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Lcorner = losses.CornerDistLoss().to(device)
    Lsml1 = nn.SmoothL1Loss()
    Sim_ssim = SSIM(data_range=255, size_average=True, channel=1)

    '''
    Validation
    '''
    eval_DistErr = utils.AverageMeter()
    eval_NCC = utils.AverageMeter()
    eval_ParaErr = utils.AverageMeter()
    eval_TransErr = utils.AverageMeter()
    eval_RotErr = utils.AverageMeter()
    eval_ParamNCC = utils.AverageMeter()
    eval_SSIM = utils.AverageMeter()
    eval_FPS = utils.AverageMeter()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    stdy_idx = 0

    with torch.no_grad():
        for data in test_loader:
            model.eval()
            # data = [t.cuda() for t in data]
            vol = data[0].cuda()
            frame = data[1].cuda()
            dof = data[3].cuda()
            for _ in range(10000):
                starter.record()
                _ = model(vol, frame, device=device)
                ender.record()
                torch.cuda.synchronize()  # 等待GPU任务完成
                curr_time = starter.elapsed_time(ender)
                eval_FPS.update(1000 / curr_time, vol.size(0))
                print(1000 / curr_time)

if __name__ == '__main__':
    '''
    GPU configuration
    '''

    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main()
