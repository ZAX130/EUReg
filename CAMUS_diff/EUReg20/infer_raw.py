import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import glob
import losses

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
from data.datasets import TrainDataset, TestDataset
from models import EUReg_FRT_Test
import matplotlib.pyplot as plt

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
    batch_size=6
    val_dir = 'Data/CAMUS2_data/Test2/'
    weights = [1, 1, 1, 1]  # loss weights
    dim=12
    layer_nums = 4
    lr = 0.0001
    scaler = [20,20,10,20,20,20]
    
    save_dir = 'raw20/'
    img_size = (32, 192, 192)
    slice_size = (128,128)

    '''
    Initialize model
    '''
    model = EUReg_FRT_Test(img_size,slice_size, dim=dim)
    model.cuda()

    '''
    DataLoader
    '''
    test_set = TestDataset(glob.glob(val_dir + '*.pkl'))
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=10, pin_memory=True)

    '''
    Criterion
    '''
    Lcorner = losses.CornerDistLoss()
    Lsml1 = nn.SmoothL1Loss()
    Sim_ssim = SSIM(data_range=1.0, size_average=True, channel=1)

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
    pic_path = 'showdata/miccai2025/CAMUS_diff/'+save_dir
    if not os.path.exists(pic_path):
        os.makedirs(pic_path)
    target_path = 'showdata/miccai2025/CAMUS_diff/target20/'
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    stdy_idx = 0
    with torch.no_grad():
        for data in test_loader:
            model.eval()
            # data = [t.cuda() for t in data]
            vol = data[0].cuda()
            frame = data[1].cuda()
            dof = data[2].cuda()

            starter.record()
            pred_dof = torch.zeros_like(dof)
            sampled_frame = model.transformer(vol, pred_dof).squeeze(2)
            ender.record()
            torch.cuda.synchronize()  # 等待GPU任务完成

            curr_time = starter.elapsed_time(ender)
            trans_l2 = losses.L2dist(pred_dof[:, :3], dof[:, :3]).item() * 0.62
            rot_l2 = losses.L2dist(pred_dof[:, 3:], dof[:, 3:]).item()
            dist_err = Lcorner(pred_dof, dof).item() * 0.62

            param_ncc = losses.transformation_parameter_NCC(pred_dof, dof).item()*100
            param_err = Lsml1(pred_dof[:, :3], dof[:, :3]).item() + Lsml1(pred_dof[:, 3:], dof[:, 3:]).item()

            ncc = losses.normalized_cross_correlation(frame, sampled_frame).item()*100
            ssim = Sim_ssim(frame, sampled_frame).item()*100

            eval_DistErr.update(dist_err, vol.size(0))
            eval_TransErr.update(trans_l2, vol.size(0))
            eval_RotErr.update(rot_l2, vol.size(0))
            eval_ParamNCC.update(param_ncc, vol.size(0))
            eval_ParaErr.update(param_err, vol.size(0))
            eval_NCC.update(ncc, vol.size(0))
            eval_SSIM.update(ssim, vol.size(0))
            if stdy_idx > 20:
                eval_FPS.update(1000 / curr_time, vol.size(0))

            print('DistErr: {:.6f}, NCC: {:.6f}, ParaErr: {:.6f}'.format(dist_err, ncc,
                                                                         param_err))
            plt.imsave(pic_path + '{:03d}_dist_{:.2f}_ncc_{:.2f}.png'.format(stdy_idx, dist_err, ncc),
                       sampled_frame.squeeze().detach().cpu().numpy(), cmap='gray')

            plt.imsave(target_path + '{:03d}_dist_{:.2f}_ncc_{:.2f}.png'.format(stdy_idx, dist_err, ncc),
                       frame.squeeze().detach().cpu().numpy(), cmap='gray')

            stdy_idx += 1
    print(save_dir)
    print('DistErr: {:.2f} +- {:.2f} mm, NCC: {:.2f} +- {:.2f} %, SSIM: {:.2f} +- {:.2f} %, '
          'TransErr: {:.2f} +- {:.2f} mm , RotErr: {:.2f} +- {:.2f}, ParamNCC: {:.2f} +- {:.2f} %, FPS: {}'.format(
        eval_DistErr.avg, eval_DistErr.std,
        eval_NCC.avg, eval_NCC.std,
        eval_SSIM.avg, eval_SSIM.std,
        eval_TransErr.avg, eval_TransErr.std,
        eval_RotErr.avg, eval_RotErr.std,
        eval_ParamNCC.avg, eval_ParamNCC.std,
        int(eval_FPS.avg)
    ))


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
