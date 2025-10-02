import numpy as np
import torch
import torch.nn as nn
import tools
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from torch.autograd import Variable

# the NCC of transformation parameters (tx, ty, tz, rx, ry, rz)
# pred_pose and target_pose are (batch_size, 6)
def transformation_parameter_NCC(pred_pose, target_pose):

    # 计算归一化互相关
    numerator = torch.sum((pred_pose - torch.mean(pred_pose, dim=1, keepdim=True)) * (target_pose - torch.mean(target_pose, dim=1, keepdim=True)), dim=1)
    denominator = torch.sqrt(torch.sum((pred_pose - torch.mean(pred_pose, dim=1, keepdim=True)) ** 2, dim=1)) * torch.sqrt(torch.sum((target_pose - torch.mean(target_pose, dim=1, keepdim=True)) ** 2, dim=1))
    ncc = torch.mean(numerator / denominator)

    return ncc

# NCC, image 1 and image 2 are (batch_size, channels, height, weight)
def normalized_cross_correlation(image1, image2):
    batch_size = image1.size(0)  # 批量大小
    image1 = image1.view(batch_size, -1)  # 转换为向量
    image2 = image2.view(batch_size, -1)  # 转换为向量

    # 计算归一化互相关
    numerator = torch.sum((image1 - torch.mean(image1, dim=1, keepdim=True)) * (image2 - torch.mean(image2, dim=1, keepdim=True)), dim=1)
    denominator = torch.sqrt(torch.sum((image1 - torch.mean(image1, dim=1, keepdim=True)) ** 2, dim=1)) * torch.sqrt(torch.sum((image2 - torch.mean(image2, dim=1, keepdim=True)) ** 2, dim=1))
    ncc = torch.mean(numerator / denominator)
    

    return ncc

def sample_slice(pred_dof, input_vol, device):
    normalize_dof = False
    dof_means = [-0.94144243, -0.51833163, -0.24158226, -0.37933836,  0.08427333, -0.01649231]
    dof_std = [11.07899716, 11.08303632, 12.56951074,  5.77583911,  5.68897645, 5.8570513]
    # original
    mat = tools.dof2mat_tensor(input_dof=pred_dof, device=device, normalize_dof = normalize_dof, dof_means= dof_means, dof_std = dof_std)
    # print('mat {}'.format(mat.shape))
    
    # print('input_vol {}'.format(input_vol.shape))
    grid = tools.myAffineGrid2(input_tensor=input_vol, input_mat=mat, 
                                input_spacing=(1, 1, 1), device=device)
    # grid = grid.to(device)
    # print('grid {}'.format(grid.shape))
    vol_resampled = F.grid_sample(input_vol, grid, align_corners=True)
    # print('resample {}'.format(vol_resampled.shape))
    # print('mat_out {}'.format(x.shape))
    slice_id = int(input_vol.shape[2] / 2)
    indices = torch.tensor([slice_id]).to(device)
    out_frame_tensor = torch.index_select(vol_resampled, 2, indices)
    return out_frame_tensor


class regularization_loss(_Loss):
    def __init__(self):
        super(regularization_loss, self).__init__(True)

        self.ms_ssim_loss = SSIM(data_range=255, size_average=True, channel=1)
        self.ms_ssim_loss_r = SSIM(data_range=255, size_average=True, channel=1)
        self.dis_err = CornerDistLoss()
        # self.prompt_loss = FocalLoss(gamma=2)
        self.prompt_loss =nn.MSELoss()
        # self.loss_t = nn.MSELoss()
        # self.loss_r = nn.MSELoss()

    def forward(self, pred_pose, target_pose, sampled_frame, input_frame, input_vol, pred_interframe_of, gt_interframe_of, seg_mask, slice_mask):
        """
        """
        
        loss_param_t = F.smooth_l1_loss(pred_pose[:,:3], target_pose[:,:3], reduction='mean')
        # loss_param_t = self.loss_t(pred_pose[:,:3], target_pose[:,:3])
        ### loss_r ####
        # pred_pose_copy = pred_pose.clone()
        # gt_pose_copy = target_pose.clone()
        # pred_pose_copy[:,0:3] = 0
        # gt_pose_copy[:,0:3] = 0
        # sampled_slice_pred = sample_slice(pred_pose_copy, input_vol, device=torch.device("cuda"))
        # sampled_slice_gt = sample_slice(gt_pose_copy, input_vol, device=torch.device("cuda"))
        # loss_param_r = 1.0 - self.ms_ssim_loss_r(sampled_slice_pred, sampled_slice_gt)

        loss_param_r = F.smooth_l1_loss(pred_pose[:,3:], target_pose[:,3:], reduction='mean')

        loss_img_similarity = 1.0 - self.ms_ssim_loss(sampled_frame, input_frame[:,0,:,:,:].unsqueeze(1).contiguous())

        transformation_pred = tools.dof6mat_tensor(pred_pose, device=torch.device("cuda"))
        transformation_true = tools.dof6mat_tensor(target_pose, device=torch.device("cuda"))
        dis_err = self.dis_err(transformation_pred,transformation_true)
        ncc_err = normalized_cross_correlation(sampled_frame, input_frame[:,0,:,:,:].unsqueeze(1).contiguous())
        # loss = loss_param + loss_img_similarity

        loss_spatial_constrain = F.smooth_l1_loss(pred_interframe_of, gt_interframe_of, reduction='mean')
        # print("seg_mask: {0}, slice_mask: {1}.".format(seg_mask.shape, slice_mask.shape) )
        loss_prompt_mask = self.prompt_loss(seg_mask, slice_mask)
        # loss_prompt_mask = F.smooth_l1_loss(seg_mask, slice_mask, reduction='mean')

        return loss_param_t, loss_param_r, loss_img_similarity, dis_err, ncc_err, loss_spatial_constrain, loss_prompt_mask

class CornerDistLoss(nn.Module):
    def __int__(self):
        super().__int__()
    def forward(self, predict, label, device=torch.device("cuda")):
        # five landmarks
        corner1 = torch.tensor([-114 / 2.0, -114 / 2.0, 0, 1], dtype=torch.float32).unsqueeze(1).to(device)
        corner2 = torch.tensor([-114 / 2.0, 114 / 2.0, 0, 1], dtype=torch.float32).unsqueeze(1).to(device)
        corner3 = torch.tensor([114 / 2.0, -114 / 2.0, 0, 1], dtype=torch.float32).unsqueeze(1).to(device)
        corner4 = torch.tensor([114 / 2.0, 114 / 2.0, 0, 1], dtype=torch.float32).unsqueeze(1).to(device)
        center = torch.tensor([0, 0, 0, 1], dtype=torch.float32).unsqueeze(1).to(device)
        # predicted points
        corner1_pred = torch.matmul(predict, corner1).squeeze(2)
        corner2_pred = torch.matmul(predict, corner2).squeeze(2)
        corner3_pred = torch.matmul(predict, corner3).squeeze(2)
        corner4_pred = torch.matmul(predict, corner4).squeeze(2)
        center_pred = torch.matmul(predict, center).squeeze(2)
        # true points
        corner1_true = torch.matmul(label, corner1).squeeze(2)
        corner2_true = torch.matmul(label, corner2).squeeze(2)
        corner3_true = torch.matmul(label, corner3).squeeze(2)
        corner4_true = torch.matmul(label, corner4).squeeze(2)
        center_true = torch.matmul(label, center).squeeze(2)

        # point error n*4 (last two entries should be zero)
        corner1_error = torch.norm(corner1_pred - corner1_true, dim = 1)
        corner2_error = torch.norm(corner2_pred - corner2_true, dim = 1)
        corner3_error = torch.norm(corner3_pred - corner3_true, dim = 1)
        corner4_error = torch.norm(corner4_pred - corner4_true, dim = 1)
        center_error = torch.norm(center_pred - center_true, dim = 1)

        loss = torch.sum(corner1_error
                        + corner2_error
                        + corner3_error
                        + corner4_error
                        + center_error) / (5.0 * center_error.shape[0])

        return loss

def get_correlation_loss(labels, outputs):
    # print('labels shape {}, outputs shape {}'.format(labels.shape, outputs.shape))
    x = outputs.flatten()
    y = labels.flatten()
    # print('x shape {}, y shape {}'.format(x.shape, y.shape))
    # print('x shape\n{}\ny shape\n{}'.format(x, y))
    xy = x * y
    mean_xy = torch.mean(xy)
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    cov_xy = mean_xy - mean_x * mean_y
    # print('xy shape {}'.format(xy.shape))
    # print('xy {}'.format(xy))
    # print('mean_xy {}'.format(mean_xy))
    # print('cov_xy {}'.format(cov_xy))

    var_x = torch.sum((x - mean_x) ** 2 / x.shape[0])
    var_y = torch.sum((y - mean_y) ** 2 / y.shape[0])
    # print('var_x {}'.format(var_x))

    corr_xy = cov_xy / (torch.sqrt(var_x * var_y))
    # print('correlation_xy {}'.format(corr_xy))

    loss = 1 - corr_xy
    # time.sleep(30)
    # x = output
    # y = target
    #
    # vx = x - torch.mean(x)
    # vy = y - torch.mean(y)
    #
    # loss = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    # print('correlation loss {}'.format(loss))
    # time.sleep(30)
    return loss


def get_dist_loss(labels, outputs, start_params, calib_mat):
    # print('labels shape {}'.format(labels.shape))
    # print('outputs shape {}'.format(outputs.shape))
    # print('start_params shape {}'.format(start_params.shape))
    # print('calib_mat shape {}'.format(calib_mat.shape))

    # print('labels_before\n{}'.format(labels.shape))
    labels = labels.data.cpu().numpy()
    outputs = outputs.data.cpu().numpy()
    if normalize_dof:
        labels = labels / dof_stats[:, 1] + dof_stats[:, 0]
        outputs = outputs / dof_stats[:, 1] + dof_stats[:, 0]


    start_params = start_params.data.cpu().numpy()
    calib_mat = calib_mat.data.cpu().numpy()

    if args.output_type == 'sum_dof':
        batch_errors = []
        for sample_id in range(labels.shape[0]):
            gen_param = tools.get_next_pos(trans_params1=start_params[sample_id, :],
                                           dof=outputs[sample_id, :],
                                           cam_cali_mat=calib_mat[sample_id, :, :])
            gt_param = tools.get_next_pos(trans_params1=start_params[sample_id, :],
                                          dof=labels[sample_id, :],
                                          cam_cali_mat=calib_mat[sample_id, :, :])
            gen_param = np.expand_dims(gen_param, axis=0)
            gt_param = np.expand_dims(gt_param, axis=0)

            result_pts = tools.params2corner_pts(params=gen_param, cam_cali_mat=calib_mat[sample_id, :, :])
            gt_pts = tools.params2corner_pts(params=gt_param, cam_cali_mat=calib_mat[sample_id, :, :])

            sample_error = tools.evaluate_dist(pts1=gt_pts, pts2=result_pts)
            batch_errors.append(sample_error)
        batch_errors = np.asarray(batch_errors)

        avg_batch_error = np.asarray(np.mean(batch_errors))
        error_tensor = torch.tensor(avg_batch_error, requires_grad=True)
        error_tensor = error_tensor.type(torch.FloatTensor)
        error_tensor = error_tensor.to(device)
        error_tensor = error_tensor * 0.99
        # print('disloss device {}'.format(device))
        # print(error_tensor)
        # time.sleep(30)
        return error_tensor




    if args.output_type == 'average_dof':
        labels = np.expand_dims(labels, axis=1)
        labels = np.repeat(labels, args.neighbour_slice - 1, axis=1)
        outputs = np.expand_dims(outputs, axis=1)
        outputs = np.repeat(outputs, args.neighbour_slice - 1, axis=1)
    else:
        labels = np.reshape(labels, (labels.shape[0], labels.shape[1] // 6, 6))
        outputs = np.reshape(outputs, (outputs.shape[0], outputs.shape[1] // 6, 6))
    # print('labels_after\n{}'.format(labels.shape))
    # print('outputs\n{}'.format(outputs.shape))
    # time.sleep(30)

    batch_errors = []
    final_drifts = []
    for sample_id in range(labels.shape[0]):
        gen_param_results = []
        gt_param_results = []
        for neighbour in range(labels.shape[1]):
            if neighbour == 0:
                base_param_gen = start_params[sample_id, :]
                base_param_gt = start_params[sample_id, :]
            else:
                base_param_gen = gen_param_results[neighbour - 1]
                base_param_gt = gt_param_results[neighbour - 1]
            gen_dof = outputs[sample_id, neighbour, :]
            gt_dof = labels[sample_id, neighbour, :]
            gen_param = tools.get_next_pos(trans_params1=base_param_gen, dof=gen_dof,
                                           cam_cali_mat=calib_mat[sample_id, :, :])
            gt_param = tools.get_next_pos(trans_params1=base_param_gt, dof=gt_dof,
                                          cam_cali_mat=calib_mat[sample_id, :, :])
            gen_param_results.append(gen_param)
            gt_param_results.append(gt_param)
        gen_param_results = np.asarray(gen_param_results)
        gt_param_results = np.asarray(gt_param_results)
        # print('gen_param_results shape {}'.format(gen_param_results.shape))

        result_pts = tools.params2corner_pts(params=gen_param_results, cam_cali_mat=calib_mat[sample_id, :, :])
        gt_pts = tools.params2corner_pts(params=gt_param_results, cam_cali_mat=calib_mat[sample_id, :, :])
        # print(result_pts.shape, gt_pts.shape)
        # time.sleep(30)

        results_final_vec = np.mean(result_pts[-1, :, :], axis=0)
        gt_final_vec = np.mean(gt_pts[-1, :, :], axis=0)
        final_drift = np.linalg.norm(results_final_vec - gt_final_vec) * 0.2
        final_drifts.append(final_drift)
        # print(results_final_vec, gt_final_vec)
        # print(final_drift)
        # time.sleep(30)

        sample_error = tools.evaluate_dist(pts1=gt_pts, pts2=result_pts)
        batch_errors.append(sample_error)

    batch_errors = np.asarray(batch_errors)
    avg_batch_error = np.asarray(np.mean(batch_errors))

    error_tensor = torch.tensor(avg_batch_error, requires_grad=True)
    error_tensor = error_tensor.type(torch.FloatTensor)
    error_tensor = error_tensor.to(device)
    error_tensor = error_tensor * 0.99
    # print('disloss device {}'.format(device))
    # print(error_tensor)
    # time.sleep(30)

    avg_final_drift = np.asarray(np.mean(np.asarray(final_drifts)))
    final_drift_tensor = torch.tensor(avg_final_drift, requires_grad=True)
    final_drift_tensor = final_drift_tensor.type(torch.FloatTensor)
    final_drift_tensor = final_drift_tensor.to(device)
    final_drift_tensor = final_drift_tensor * 0.99
    return error_tensor, final_drift_tensor


class FocalLoss(_Loss):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            # print("fcls input.size", input.size(), target.size())
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)
        # print("fcls reshape input.size", input.size(), target.size())

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()