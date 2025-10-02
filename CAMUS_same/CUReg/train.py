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
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default = 'CAMUS', help='')
parser.add_argument('--dataset_root', type=str, default = '', help='dataset root dir ()')
parser.add_argument('--batch_size', type=int, default = 12, help='batch size')
parser.add_argument('--workers', type=int, default = 10, help='number of data loading workers')
parser.add_argument('--lr', default=0.00005, help='learning rate')
parser.add_argument('--lr_rate', default=0.3, help='learning rate decay rate')
parser.add_argument('--w', default=0.015, help='learning rate')
parser.add_argument('--w_rate', default=0.3, help='learning rate decay rate')
parser.add_argument('--decay_margin', default=0.016, help='margin to decay lr & w')
parser.add_argument('--noise_trans', default=0.03, help='range of the random noise of translation added to the training data')
parser.add_argument('--iteration', type=int, default = 2, help='number of refinement iterations')
parser.add_argument('--nepoch', type=int, default=500, help='max number of epochs to train')
parser.add_argument('--resume_posenet', type=str, default = '',  help='resume PoseNet model')
parser.add_argument('--start_epoch', type=int, default = 0, help='which epoch to start')
parser.add_argument('--use_img_similarity', type=bool, default =True, help='')
opt = parser.parse_args()

def ensure_fd(fd):
    if not os.path.exists(fd):
        os.system('mkdir -p {}'.format(fd))
torch.backends.cudnn.benchmark=True
def main():
    opt.manualSeed = random.randint(1, 10000)
    print(opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.dataset == 'CAMUS':
        opt.outf = 'experiments/' + opt.dataset+'/trained_models' #folder to save trained models
        opt.log_dir = 'logs/' + opt.dataset + '/logtxt'#folder to save logs
        opt.train_info_dir = 'logs/' + opt.dataset + '/train_info' #folder to save logs
        opt.repeat_epoch = 2 #number of repeat times for one epoch training
        opt.dataset_root = 'Data/CAMUS'
        makedirs(opt.outf)
        makedirs(opt.log_dir)
        makedirs(opt.train_info_dir)
    else:
        print('Unknown dataset')
        return
    
    ensure_fd(opt.outf)
    ensure_fd(opt.log_dir)
    ensure_fd(opt.train_info_dir)

    writer = SummaryWriter(log_dir=opt.train_info_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # estimator = PoseNet(num_points = opt.num_points, num_obj = opt.num_objects)
    estimator = RegistNetwork(layers=[3, 8, 36, 3])
    estimator.cuda()
    
    if opt.resume_posenet != '':
        print("resume_posenet: ", opt.resume_posenet)
        estimator.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_posenet)))
    
    opt.decay_start = False
    optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)
    
    dataset = PoseDataset_s2v('train', opt.dataset_root)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers, drop_last=True)
    test_dataset = PoseDataset_s2v('test', opt.dataset_root)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers)

    print('>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the training set: {0}\nlength of the testing set: {1}\n'.format(len(dataset), len(test_dataset)))

    criterion = regularization_loss().to(device)


    best_test = np.Inf 
    
    
    if opt.start_epoch == 1:
        for log in os.listdir(opt.log_dir):
            os.remove(os.path.join(opt.log_dir, log))
    st_time = time.time()

    it = 0

    for epoch in range(opt.start_epoch, opt.nepoch): 
        
        logger = setup_logger('epoch%d' % epoch, os.path.join(opt.log_dir, 'epoch_%d_log.txt' % epoch))
        logger.info('Train time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Training started'))
        train_count = 0 
        train_dis_avg = 0.0
        train_ncc_avg = 0.0
        train_param_t_avg = 0.0
        train_param_r_avg = 0.0
        train_sim_avg = 0.0
        train_spatial_constrain_avg = 0.0
        train_prompt_mask_avg=0.0
        
        
        estimator.train()
        optimizer.zero_grad() 

        for rep in range(opt.repeat_epoch): 
            for i, data in enumerate(dataloader, 0):
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
                # print("dof_tensor: {0}, dof_6_pred: {1}.".format(dof_tensor, dof_6_pred) )
                
                loss_param_t, loss_param_r, loss_img_similarity, dis_err, ncc_err, loss_spatial_constrain, loss_prompt_mask = criterion.forward(
                    dof_6_pred, dof_tensor, resampled_frame_full, frame_tensor, vol_tensor, pred_interframe_of, relative_trans, seg_mask,slice_mask)
                
                if opt.use_img_similarity:
                    loss_lst = [(loss_param_t, 1.0), (loss_param_r, 1.0), (loss_spatial_constrain, 0.1), (loss_img_similarity, 0.5), (loss_prompt_mask, 0.1)]
                    loss = sum([ls * w for ls, w in loss_lst])

                    loss_dict = {
                    'loss_r': loss_param_r.item(),
                    'loss_t':loss_param_t.item(),
                    'loss_spatial_constrain':loss_spatial_constrain.item(),
                    'loss_img_similarity':loss_img_similarity.item(),
                    'loss_prompt_mask':loss_prompt_mask.item(),
                    'loss_all': loss.item()
                }
                    
                    writer.add_scalars('loss', loss_dict, it)
                else:
                    loss_lst = [(loss_param_t, 1.0), (loss_param_r, 2.0), (loss_spatial_constrain, 1.0),  (loss_prompt_mask, 2.0)]
                    loss = sum([ls * w for ls, w in loss_lst])

                    loss_dict = {
                    'loss_r': loss_param_r.item(),
                    'loss_t':loss_param_t.item(),
                    'loss_spatial_constrain':loss_spatial_constrain.item(),
                    'loss_prompt_mask':loss_prompt_mask.item(),
                    'loss_all': loss.item()}
                    
                    writer.add_scalars('loss', loss_dict, it)
                
                loss.backward()

                train_dis_avg += dis_err.item()
                train_ncc_avg += ncc_err.item()
                train_param_t_avg += loss_param_t.item()
                train_param_r_avg += loss_param_r.item()
                train_spatial_constrain_avg += loss_spatial_constrain.item()
                train_prompt_mask_avg += loss_prompt_mask.item()
                train_sim_avg += loss_img_similarity.item()
                train_count += 1
                it += 1

                optimizer.step()
                optimizer.zero_grad()
                print('Iter {} DistErr {:.6f} NCCErr {:.6f} T {:.6f} R {:.6f} SC {:.6f} Prompt {:.6f} Sim {:.6f}'.format(
                    it, dis_err.item(), ncc_err.item(), loss_param_t.item(), loss_param_r.item(), loss_spatial_constrain.item(),
                    loss_prompt_mask.item(), loss_img_similarity.item()
                ))

                if train_count % opt.batch_size == 0:
                    logger.info('Train_time {0} Epoch {1} Loss_all {2} Avg_t_Err:{3} Avg_r_Err:{4} Avg_sim_Err:{5} Avg_dis:{6} Avg_ncc:{7} Avg_spatial:{8} Avg_prompt:{9}'
                                .format('Nan', epoch, 
                                loss, train_param_t_avg / opt.batch_size, 
                                train_param_r_avg / opt.batch_size, train_sim_avg / opt.batch_size, train_dis_avg / opt.batch_size, 
                                train_ncc_avg / opt.batch_size, train_spatial_constrain_avg / opt.batch_size, train_prompt_mask_avg / opt.batch_size))
                    train_dis_avg = 0
                    train_param_t_avg=0
                    train_param_r_avg=0
                    train_sim_avg =0
                    train_ncc_avg=0
                    train_spatial_constrain_avg=0
                    train_prompt_mask_avg=0



                if train_count != 0 and train_count % 1000 == 0:
                    torch.save(estimator.state_dict(), '{0}/pose_model_current.pth'.format(opt.outf))

        print('>>>>>>>>----------epoch {0} train finish---------<<<<<<<<'.format(epoch))

        
        logger = setup_logger('epoch%d_test' % epoch, os.path.join(opt.log_dir, 'epoch_%d_test_log.txt' % epoch))
        logger.info('Test time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Testing started'))
        test_dis = 0.0
        test_ncc = 0.0
        test_count = 0
        
        estimator.eval()
        
        
        for j, data in enumerate(testdataloader, 0):
            case_id, vol_tensor, frame_tensor, mat_tensor, dof_tensor, relative_trans, slice_mask = data
            case_id, vol_tensor, frame_tensor, mat_tensor, dof_tensor, relative_trans, slice_mask = case_id.cuda(), \
                                                                 Variable(vol_tensor).cuda(), \
                                                                 Variable(frame_tensor).cuda(), \
                                                                 Variable(mat_tensor).cuda(), \
                                                                 Variable(dof_tensor).cuda(), \
                                                                 Variable(relative_trans).cuda(), \
                                                                 Variable(slice_mask).cuda(), \
            
            vol_resampled, dof_6_pred, resampled_frame, pred_interframe_of, seg_mask = estimator(vol_tensor, frame_tensor, device=device)
        
            loss_param_t, loss_param_r, loss_img_similarity, dis_err, ncc_err, loss_spatial_constrain, loss_prompt_mask = criterion.forward(
                dof_6_pred, dof_tensor, resampled_frame, frame_tensor,vol_tensor,  pred_interframe_of, relative_trans,seg_mask,slice_mask)
                
            test_para = loss_param_t.item() + loss_param_r.item()
            test_dis += test_para
            # test_dis += dis_err.item()
            test_ncc += ncc_err.item()
            logger.info('Test time {0} Test Frame No.{1} dis:{2} ncc:{3} param_loss:{4}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), test_count, dis_err, ncc_err,test_para))

            test_count += 1

        test_dis = test_dis / test_count
        test_ncc = test_ncc / test_count

        val_dict={}
        val_dict['dis_err'] = test_dis
        val_dict['Ncc'] = test_ncc
        writer.add_scalars('val_acc', val_dict, it)

        
        logger.info('Test time {0} Epoch {1} TEST FINISH Avg dis: {2} Avg ncc: {3}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, test_dis, test_ncc))
        if test_dis <= best_test:
            best_test = test_dis
            
            torch.save(estimator.state_dict(), '{0}/pose_model_{1}_{2}.pth'.format(opt.outf, epoch, test_dis))
            print(epoch, '>>>>>>>>----------BEST TEST MODEL SAVED---------<<<<<<<<')

        if best_test < opt.decay_margin and not opt.decay_start:
            opt.decay_start = True
            opt.lr *= opt.lr_rate
            opt.w *= opt.w_rate
            optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)
    
    writer.close()
if __name__ == '__main__':
    main()
