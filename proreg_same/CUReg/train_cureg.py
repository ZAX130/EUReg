import glob
import os, losses, utils
import sys
from torch.utils.data import DataLoader
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from data.datasets import TrainDataset_postcut, TestDataset
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from natsort import natsorted
from networks.dual_fusionNet import RegistNetwork
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

    train_dir = 'Data/proregus2_data/Train/'
    val_dir = 'Data/proregus2_data/Test5_10_post64/'
    weights = [1, 1, 0, 1, 0]  # loss weights
    lr = 0.00005
    scaler = [5,5,10,10,10,10]
    save_dir = 'CUReg/'.format(*weights, batch_size, lr)
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
    model = RegistNetwork(layers=[3, 8, 36, 3])
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
    Lsml1 = nn.SmoothL1Loss()
    Lssim = losses.SSIMLoss()
    Lmse = nn.MSELoss()
    Lcorner = losses.CornerDistLoss()

    best_ParaErr = 100
    it = 0
    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')
        '''
        Training
        '''
        loss_all = utils.AverageMeter()
        idx = 0
        for data in train_loader:
            model.train()
            idx += 1
            it += 1
            # adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            # data = [t.cuda() for t in data]
            vol = data[0].cuda()
            frame = data[1].cuda()
            # mask = data[2]
            dof = data[2].cuda()
            # frame_dist = data[4]

            pred_dof, sampled_frame = model(vol, frame, device=device)

            l_trl = Lsml1(pred_dof[:, :3], dof[:, :3])
            l_rot = Lsml1(pred_dof[:, 3:], dof[:, 3:])
            l_sim = Lssim(frame.squeeze(), sampled_frame.squeeze())
            loss = l_trl*weights[0] + l_rot*weights[1] + l_sim*weights[3] # + l_dist*weights[2] + l_sim*weights[3] + l_seg*weights[4]
            dist_err = Lcorner(pred_dof, dof)
            loss_all.update(loss.item(), vol.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('{} TotelIter {} loss {:.6f}, DistErr {:.6f}, T: {:.6f}, R: {:.6f}'.format(epoch, it,
                                                                                                                loss.item(),
                                                                                                                dist_err.item(),
                                                                                                                l_trl.item(),
                                                                                                                l_rot.item(),

                                                                                                                ))
        if (epoch+1) % 30 == 0 or epoch>max_epoch-5:
            '''
            Validation
            '''
            eval_DistErr = utils.AverageMeter()
            eval_NCC = utils.AverageMeter()
            eval_ParaErr = utils.AverageMeter()
            with torch.no_grad():
                for data in val_loader:
                    model.eval()
                    # data = [t.cuda() for t in data]
                    vol = data[0].cuda()
                    frame = data[1].cuda()
                    dof = data[2].cuda()

                    pred_dof, sampled_frame = model(vol, frame, device=device)

                    param_err = Lsml1(pred_dof[:, :3], dof[:, :3]).item() + Lsml1(pred_dof[:, 3:],
                                                                                  dof[:, 3:]).item()
                    dist_err = Lcorner(pred_dof, dof).item()
                    ncc = losses.normalized_cross_correlation(frame, sampled_frame).item()

                    eval_ParaErr.update(param_err, vol.size(0))
                    eval_DistErr.update(dist_err, vol.size(0))
                    eval_NCC.update(ncc, vol.size(0))

                    print('Epoch {} DistErr: {:.6f}, NCC: {:.6f}, ParaErr: {:.6f}'.format(epoch, dist_err, ncc,
                                                                                          param_err))
            best_ParaErr = min(eval_ParaErr.avg, best_ParaErr)
            print(save_dir)
            print('Epoch', epoch, 'DistErr:', eval_DistErr.avg, 'NCC:', eval_NCC.avg, 'ParaErr:', eval_ParaErr.avg)
            print(epoch, eval_DistErr.avg, eval_NCC.avg, eval_ParaErr.avg, file=f)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_DistErr': best_ParaErr,
                'optimizer': optimizer.state_dict(),
            }, save_dir='experiments/proregus2/' + save_dir,
                filename='ParaErr{:.4f}_DistErr{:.4f}_epoch{}.pth.tar'.format(eval_ParaErr.avg, eval_DistErr.avg,
                                                                              epoch))
    print('{} Epoch {} loss {:.4f}'.format(save_dir, epoch, loss_all.avg))
    print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg), file=f, end=' ')

    loss_all.reset()

def comput_fig(img):
    img = img.detach().cpu().numpy()[0, 0, 48:64, :, :]
    fig = plt.figure(figsize=(12, 12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)

def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 192, 160)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j+line_thickness-1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i+line_thickness-1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img

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