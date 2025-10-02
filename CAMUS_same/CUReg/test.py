import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import random
import time
import numpy as np
import torch
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
import tools
import torch.nn.functional as F
import cv2
from lib.loss import transformation_parameter_NCC
from batchgenerators.utilities.file_and_folder_operations import makedirs
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default = 'CAMUS', help='')
parser.add_argument('--dataset_root', type=str, default = '', help='dataset root dir ()')
parser.add_argument('--batch_size', type=int, default = 1, help='batch size')
parser.add_argument('--workers', type=int, default = 10, help='number of data loading workers')
parser.add_argument('--results', type=str, default = '', help='dataset root dir')
parser.add_argument('--model', type=str, default = 'trained_model_CU-reg.pth',  help='resume PoseNet model')
parser.add_argument('--use_img_similarity', type=bool, default =True, help='')
opt = parser.parse_args()

def ensure_fd(fd):
    if not os.path.exists(fd):
        os.system('mkdir -p {}'.format(fd))

def main():
    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if opt.dataset == 'CAMUS':

        opt.results = 'experiments/results/' + opt.dataset  #folder to save logs
        opt.outf = 'experiments/miccai2025/CAMUS_same/CUReg' #folder to save trained models
        opt.log_dir = 'logs/' + opt.dataset + '/logtxt'#folder to save logs
        opt.train_info_dir = 'logs/' + opt.dataset + '/train_info' #folder to save logs
        opt.repeat_epoch = 2 #number of repeat times for one epoch training
        opt.dataset_root = 'Data/CAMUS'

        makedirs(opt.results)
        makedirs(opt.log_dir)
        makedirs(opt.train_info_dir)
        makedirs(opt.dataset_root)
    else:
        print('Unknown dataset')
        return
    
    # ensure_fd(opt.outf)
    # ensure_fd(opt.log_dir)
    # ensure_fd(opt.results)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # estimator = PoseNet(num_points = opt.num_points, num_obj = opt.num_objects)
    estimator = RegistNetwork(layers=[3, 8, 36, 3])
    
    
    
    
    print("model: '{0}/{1}".format(opt.outf, opt.model))
        
    estimator.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.model)))
    print("Checkpoint loaded.")
    estimator = estimator.cuda()
    
    test_dataset = PoseDataset_s2v('test', opt.dataset_root)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0)

    print('>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the testing set: {0}\n'.format(len(test_dataset)))

    criterion = regularization_loss().to(device)

    
    
    st_time = time.time()

    #保存每次测试的log文件
    logger = setup_logger('test', os.path.join(opt.log_dir, 'epoch_test_log.txt'))
    logger.info('Test time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Testing started'))
    test_dis = 0.0
    test_ncc = 0.0
    test_sim = 0.0
    test_count = 0
    t_err = 0.0
    r_err = 0.0
    ncc_param = 0.0
    
    estimator.eval()
    
    
    with torch.no_grad():
        for j, data in enumerate(testdataloader, 0):
            case_id, vol_tensor, frame_tensor, mat_tensor, dof_tensor, relative_trans, slice_mask = data
            # print("case_id: ", case_id)
            # print("vol_tensor: ", vol_tensor.shape)
            case_id, vol_tensor, frame_tensor, mat_tensor, dof_tensor, relative_trans, slice_mask = case_id.cuda(), \
                                                                Variable(vol_tensor).cuda(), \
                                                                Variable(frame_tensor).cuda(), \
                                                                Variable(mat_tensor).cuda(), \
                                                                Variable(dof_tensor).cuda(), \
                                                                Variable(relative_trans).cuda(), \
                                                                Variable(slice_mask).cuda(), \
            
            vol_resampled, dof_6_pred, resampled_frame_full, pred_interframe_of, seg_mask = estimator(vol_tensor, frame_tensor, device=device)
        
            loss_param_t, loss_param_r, loss_img_similarity, dis_err, ncc_err, loss_spatial_constrain, loss_prompt_mask = criterion.forward(
                    dof_6_pred, dof_tensor, resampled_frame_full, frame_tensor, vol_tensor, pred_interframe_of, relative_trans, seg_mask,slice_mask)
            
                
            test_dis += dis_err.item()*0.62
            test_ncc += ncc_err.item()
            test_sim += (1 - loss_img_similarity.item())
            
            translation_err = torch.mean(torch.norm((dof_6_pred[:,:3] - dof_tensor[:,:3]),p=2))*0.62
            rotation_err = torch.mean(torch.norm((dof_6_pred[:,3:] - dof_tensor[:,3:]), p=2))
            t_err += translation_err
            r_err += rotation_err
            ncc_param += transformation_parameter_NCC(dof_6_pred, dof_tensor)
            

            logger.info('Test time {0} Test Frame No.{1} dis:{2} ncc:{3}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), test_count, dis_err, ncc_err))

            test_count += 1

            ##### vis ###
            slice_id = int(vol_tensor.shape[2] / 2)
            indices = torch.tensor([slice_id]).to(device)
            out_frame_tensor = torch.index_select(vol_resampled, 2, indices).squeeze()
            # out_frame_tensor = vol_resampled[:, :, slice_id, :, :].squeeze()
            frame_tensor = frame_tensor.squeeze()
            mat_tensor_convert = tools.dof2mat_tensor(input_dof=dof_tensor, device=device, normalize_dof = False)
            grid = tools.myAffineGrid2(input_tensor=vol_tensor, input_mat=mat_tensor_convert,
                            input_spacing=(1, 1, 1), device=device)
            grid = grid.to(device)
            gt_resampled = F.grid_sample(vol_tensor, grid, align_corners=True)
            gt_np = gt_resampled[:, :, slice_id, :, :].squeeze().cpu().detach().numpy().copy()

            out_np = out_frame_tensor.cpu().detach().squeeze().numpy()
            frame_np = frame_tensor.cpu().squeeze().numpy()
            if out_np.ndim > 2:
                out_np = out_np[0, :, :]
                gt_np = gt_np[0, :, :]
                frame_np = frame_np[0, :, :]
            # cat_np = np.concatenate((frame_np, gt_np, out_np), axis=0)
            residual_img = abs(frame_np - out_np)
            cat_np = np.concatenate((frame_np, out_np, residual_img), axis=0)
            
            # val_results_dir = os.path.join(opt.results, '{}_{}'.format(run_num, phase))
            # if not os.path.exists(val_results_dir):
            #     os.makedirs(val_results_dir)
            cv2.imwrite(os.path.join(opt.results, 'frame_{}.jpg'.format(j)), cat_np)
            # cv2.imwrite(os.path.join(opt.results, 'frame_residual_{}.jpg'.format(j)), residual_img)
            # record the true and predicted parameters
            dof_label = dof_tensor[0, :].cpu().detach().unsqueeze(0).numpy()
            dof_pred = dof_6_pred[0, :].cpu().detach().unsqueeze(0).numpy()
            dof_compare = np.concatenate((dof_label, dof_pred), axis=0).astype(float)

            dof_compare_dir = os.path.join(opt.results, 'paramCompare_{}.txt'.format(j))
            np.savetxt(dof_compare_dir, dof_compare, fmt='%f')

    test_dis = test_dis / test_count
    test_ncc = test_ncc / test_count
    test_sim = test_sim / test_count
    t_err = t_err / test_count
    r_err = r_err / test_count
    ncc_param = ncc_param / test_count
    
    logger.info('Runtime {0} Avg dis: {1} Avg ncc: {2} Avg ssim: {3} Avg trans: {4} Avg rot: {5} Avg param_ncc: {6}'
                .format((time.time() - st_time) / test_count, test_dis, test_ncc * 100.0, test_sim * 100.0, t_err, r_err, ncc_param * 100.0))

if __name__ == '__main__':
    main()
