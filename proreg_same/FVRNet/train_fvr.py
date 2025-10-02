import glob
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import losses, utils
import sys
from torch.utils.data import DataLoader
from data.datasets import TrainDataset_postcut, TestDataset
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from natsort import natsorted
from networks.fvrnet import mynet3
import random
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
    torch.backends.cudnn.benchmark=True

same_seeds(2032)

class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir+"logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
GPU_iden = 0
def main():
    batch_size = 6
    # tools networks.tools:dof2matTensor, dof6mat_Tensor恢复梯度
    # CUOriTrainDataset TestDataset体积和每个切片都归一化
    # PSPNet seg分支修改为单通道sigmoid输出
    # MS-SSIM data_range修改为1.0
    # self.convdown_prompt_1 输入通道修改为1
    train_dir = 'Data/proregus2_data/Train/'
    val_dir = 'Data/proregus2_data/Test5_10_post64/'
    weights = [1,1]  # loss weights
    lr = 0.00005
    scaler = [5, 5, 10, 10, 10, 10]
    save_dir = 'FVRNet/'
    img_size = (40, 64, 64)
    slice_size = (64,64)
    if not os.path.exists('experiments/proregus2/' + save_dir):
        os.makedirs('experiments/proregus2/' + save_dir)
    if not os.path.exists('logs/' + save_dir):
        os.makedirs('logs/' + save_dir)
    sys.stdout = Logger('logs/' + save_dir)
    f = open(os.path.join('logs/'+save_dir, 'losses' + ".txt"), "a")
    device = torch.device("cuda")

    epoch_start = 0
    max_epoch = 30000
    cont_training = False

    '''
    Initialize model
    '''
    model = mynet3(layers=[3, 8, 36, 3])
    model.cuda()


    '''
    If continue from previous training
    '''
    if cont_training:
        # epoch_start = 384
        model_dir = 'experiments/proregus2/'+save_dir
        updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch,0.9),8)
        best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[-1])['state_dict']
        model.load_state_dict(best_model)
        print(model_dir + natsorted(os.listdir(model_dir))[-1])
    else:
        updated_lr = lr

    '''
    Initialize training
    '''

    train_set = TrainDataset_postcut(glob.glob(train_dir + '*.pkl'), img_size, slice_size, scaler, lower_bound=0.05)
    val_set = TestDataset(glob.glob(val_dir + '*.pkl'))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=10, pin_memory=True)

    optimizer = optim.Adam(model.parameters(), lr=updated_lr)

    Lmse = nn.MSELoss()
    Lcorner = losses.CornerDistLoss()
    Lsml1 = nn.SmoothL1Loss()
    best_ParaErr = 100
    it = 0
    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')
        '''
        Training
        '''
        loss_all = utils.AverageMeter()
        idx = 0
        model.train()
        for data in train_loader:
            idx += 1
            it += 1
            # adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            data = [t.cuda() for t in data]
            vol = data[0]
            frame = data[1]
            dof = data[2]

            _, pred_dof, sampled_frame = model(vol, frame, device=device)

            l_param = Lmse(pred_dof, dof)
            l_sim = Lmse(sampled_frame, frame)
            loss = l_param*weights[0] + l_sim*weights[1]
            dist_err = Lcorner(pred_dof, dof)
            loss_all.update(loss.item(), vol.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('{} TotelIter {} loss {:.6f}, DistErr {:.6f}, Param: {:.6f}, Sim: {:.6f}'.format(epoch, it,
                                                                                                                loss.item(),
                                                                                                                dist_err.item(),
                                                                                                                l_param.item(),
                                                                                                                l_sim.item(),
                                                                                                                ))

        print('{} Epoch {} loss {:.4f}'.format(save_dir, epoch, loss_all.avg))
        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg), file=f, end=' ')
        if (epoch+1) % 30 == 0 or epoch > max_epoch-5:
            '''
            Validation
            '''
            eval_DistErr = utils.AverageMeter()
            eval_NCC = utils.AverageMeter()
            eval_ParaErr = utils.AverageMeter()
            with torch.no_grad():
                for data in val_loader:
                    model.eval()
                    data = [t.cuda() for t in data]
                    vol = data[0]
                    frame = data[1]
                    dof = data[2]

                    _, pred_dof, sampled_frame = model(vol, frame, device=device)

                    param_err = Lsml1(pred_dof[:, :3], dof[:, :3]).item() + Lsml1(pred_dof[:, 3:], dof[:, 3:]).item()
                    dist_err = Lcorner(pred_dof, dof).item()
                    ncc = losses.normalized_cross_correlation(frame, sampled_frame).item()

                    eval_ParaErr.update(param_err, vol.size(0))
                    eval_DistErr.update(dist_err, vol.size(0))
                    eval_NCC.update(ncc, vol.size(0))

                    print('Epoch {} DistErr: {:.6f}, NCC: {:.6f}, ParaErr: {:.6f}'.format(epoch, dist_err, ncc,param_err))
            best_ParaErr = min(eval_ParaErr.avg, best_ParaErr)
            print(save_dir)
            print('Epoch', epoch, 'DistErr:', eval_DistErr.avg, 'NCC:', eval_NCC.avg, 'ParaErr:', eval_ParaErr.avg)
            print(epoch, eval_DistErr.avg, eval_NCC.avg, eval_ParaErr.avg, file=f)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_DistErr': best_ParaErr,
                'optimizer': optimizer.state_dict(),
            }, save_dir='experiments/proregus2/' + save_dir, filename='ParaErr{:.4f}_epoch{}.pth.tar'.format(eval_ParaErr.avg, epoch))
            loss_all.reset()

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)

def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=20):
    torch.save(state, save_dir+filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[-1])
        model_lists = natsorted(glob.glob(save_dir + '*'))

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