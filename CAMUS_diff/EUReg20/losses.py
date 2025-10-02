import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from torch.autograd import Variable
import torchgeometry as tgm
import math


# the NCC of transformation parameters (tx, ty, tz, rx, ry, rz)
# pred_pose and target_pose are (batch_size, 6)
def transformation_parameter_NCC(pred_pose, target_pose):
    # 计算归一化互相关
    numerator = torch.sum((pred_pose - torch.mean(pred_pose, dim=1, keepdim=True)) * (
            target_pose - torch.mean(target_pose, dim=1, keepdim=True)), dim=1)
    denominator = torch.sqrt(
        torch.sum((pred_pose - torch.mean(pred_pose, dim=1, keepdim=True)) ** 2, dim=1)) * torch.sqrt(
        torch.sum((target_pose - torch.mean(target_pose, dim=1, keepdim=True)) ** 2, dim=1))
    ncc = torch.mean(numerator / denominator)

    return ncc

def L2dist(pred_pose, target_pose):
    return torch.norm(pred_pose - target_pose, p=2).mean()

# NCC, image 1 and image 2 are (batch_size, channels, height, weight)
def normalized_cross_correlation(image1, image2):
    batch_size = image1.size(0)  # 批量大小
    image1 = image1.view(batch_size, -1)  # 转换为向量
    image2 = image2.view(batch_size, -1)  # 转换为向量

    # 计算归一化互相关
    numerator = torch.sum(
        (image1 - torch.mean(image1, dim=1, keepdim=True)) * (image2 - torch.mean(image2, dim=1, keepdim=True)), dim=1)
    denominator = torch.sqrt(torch.sum((image1 - torch.mean(image1, dim=1, keepdim=True)) ** 2, dim=1)) * torch.sqrt(
        torch.sum((image2 - torch.mean(image2, dim=1, keepdim=True)) ** 2, dim=1))
    ncc = torch.mean(numerator / denominator)

    return ncc


class SSIMLoss(torch.nn.Module):
    def __init__(self, data_range=255):
        super(SSIMLoss, self).__init__()
        self.ms_ssim_loss = SSIM(data_range=data_range, size_average=True, channel=1)

    def forward(self, input, target):
        return 1.0 - self.ms_ssim_loss(input, target)


# class regularization_loss(_Loss):
#     def __init__(self):
#         super(regularization_loss, self).__init__(True)
#
#         self.ms_ssim_loss = SSIM(data_range=255, size_average=True, channel=1)
#         self.ms_ssim_loss_r = SSIM(data_range=255, size_average=True, channel=1)
#         self.dis_err = CornerDistLoss()
#         # self.prompt_loss = FocalLoss(gamma=2)
#         self.prompt_loss =nn.MSELoss()
#         # self.loss_t = nn.MSELoss()
#         # self.loss_r = nn.MSELoss()
#
#     def forward(self, pred_pose, target_pose, sampled_frame, input_frame, input_vol, pred_interframe_of, gt_interframe_of, seg_mask, slice_mask):
#         """
#         """
#
#         loss_param_t = F.smooth_l1_loss(pred_pose[:,:3], target_pose[:,:3], reduction='mean')
#         # loss_param_t = self.loss_t(pred_pose[:,:3], target_pose[:,:3])
#         ### loss_r ####
#         # pred_pose_copy = pred_pose.clone()
#         # gt_pose_copy = target_pose.clone()
#         # pred_pose_copy[:,0:3] = 0
#         # gt_pose_copy[:,0:3] = 0
#         # sampled_slice_pred = sample_slice(pred_pose_copy, input_vol, device=torch.device("cuda"))
#         # sampled_slice_gt = sample_slice(gt_pose_copy, input_vol, device=torch.device("cuda"))
#         # loss_param_r = 1.0 - self.ms_ssim_loss_r(sampled_slice_pred, sampled_slice_gt)
#
#         loss_param_r = F.smooth_l1_loss(pred_pose[:,3:], target_pose[:,3:], reduction='mean')
#
#         loss_img_similarity = 1.0 - self.ms_ssim_loss(sampled_frame, input_frame[:,0,:,:,:].unsqueeze(1).contiguous())
#
#         transformation_pred = utils.dof6mat_tensor(pred_pose, device=torch.device("cuda"))
#         transformation_true = utils.dof6mat_tensor(target_pose, device=torch.device("cuda"))
#         dis_err = self.dis_err(transformation_pred,transformation_true)
#         ncc_err = normalized_cross_correlation(sampled_frame, input_frame[:,0,:,:,:].unsqueeze(1).contiguous())
#         # loss = loss_param + loss_img_similarity
#
#         loss_spatial_constrain = F.smooth_l1_loss(pred_interframe_of, gt_interframe_of, reduction='mean')
#         # print("seg_mask: {0}, slice_mask: {1}.".format(seg_mask.shape, slice_mask.shape) )
#         loss_prompt_mask = self.prompt_loss(seg_mask, slice_mask)
#         # loss_prompt_mask = F.smooth_l1_loss(seg_mask, slice_mask, reduction='mean')
#
#         return loss_param_t, loss_param_r, loss_img_similarity, dis_err, ncc_err, loss_spatial_constrain, loss_prompt_mask

# class CornerDistLoss(nn.Module):
#     def __int__(self, corner_h=114, corner_w=114):
#         super().__int__()
#
#         corner1 = torch.tensor([-corner_h / 2.0, -corner_w / 2.0, 0, 1], dtype=torch.float32).unsqueeze(1)
#         corner2 = torch.tensor([-corner_h / 2.0, corner_w / 2.0, 0, 1], dtype=torch.float32).unsqueeze(1)
#         corner3 = torch.tensor([corner_h / 2.0, -corner_w / 2.0, 0, 1], dtype=torch.float32).unsqueeze(1)
#         corner4 = torch.tensor([corner_h / 2.0, corner_w / 2.0, 0, 1], dtype=torch.float32).unsqueeze(1)
#         center = torch.tensor([0, 0, 0, 1], dtype=torch.float32).unsqueeze(1)
#
#         self.register_buffer('corner1', corner1)
#         self.register_buffer('corner2', corner2)
#         self.register_buffer('corner3', corner3)
#         self.register_buffer('corner4', corner4)
#         self.register_buffer('center', center)
#
#     def dof2mat(self, input_dof):
#         rad = tgm.deg2rad(input_dof[:, 3:])
#
#         ai, aj, ak = rad[:, 0], rad[:, 1], rad[:, 2]
#         si, sj, sk = torch.sin(ai), torch.sin(aj), torch.sin(ak)
#         ci, cj, ck = torch.cos(ai), torch.cos(aj), torch.cos(ak)
#         cc, cs = ci * ck, ci * sk
#         sc, ss = si * ck, si * sk
#         M = torch.eye(4, dtype=input_dof.dtype, device=input_dof.device).repeat(input_dof.shape[0], 1, 1)
#         M[:, 0, 0] = cj * ck
#         M[:, 0, 1] = sj * sc - cs
#         M[:, 0, 2] = sj * cc + ss
#         M[:, 1, 0] = cj * sk
#         M[:, 1, 1] = sj * ss + cc
#         M[:, 1, 2] = sj * cs - sc
#         M[:, 2, 0] = -sj
#         M[:, 2, 1] = cj * si
#         M[:, 2, 2] = cj * ci
#         M[:, :3, 3] = input_dof[:, :3]  # 平移分量
#
#         return M
#     def forward(self, pred_pose, target_pose):
#         # [Th, Tw, Td, Rh, Rw, Rd]
#         predict = self.dof2mat(pred_pose)
#         label = self.dof2mat(target_pose)
#
#         # predicted points
#         corner1_pred = torch.matmul(predict, self.corner1).squeeze(2)
#         corner2_pred = torch.matmul(predict, self.corner2).squeeze(2)
#         corner3_pred = torch.matmul(predict, self.corner3).squeeze(2)
#         corner4_pred = torch.matmul(predict, self.corner4).squeeze(2)
#         center_pred = torch.matmul(predict, self.center).squeeze(2)
#         # true points
#         corner1_true = torch.matmul(label, self.corner1).squeeze(2)
#         corner2_true = torch.matmul(label, self.corner2).squeeze(2)
#         corner3_true = torch.matmul(label, self.corner3).squeeze(2)
#         corner4_true = torch.matmul(label, self.corner4).squeeze(2)
#         center_true = torch.matmul(label, self.center).squeeze(2)
#
#         # point error n*4 (last two entries should be zero)
#         corner1_error = torch.norm(corner1_pred - corner1_true, dim = 1)
#         corner2_error = torch.norm(corner2_pred - corner2_true, dim = 1)
#         corner3_error = torch.norm(corner3_pred - corner3_true, dim = 1)
#         corner4_error = torch.norm(corner4_pred - corner4_true, dim = 1)
#         center_error = torch.norm(center_pred - center_true, dim = 1)
#
#         loss = torch.sum(corner1_error
#                         + corner2_error
#                         + corner3_error
#                         + corner4_error
#                         + center_error) / (5.0 * center_error.shape[0])
#
#         return loss
class CornerDistLoss(nn.Module):
    def __int__(self):
        super().__int__()

    @torch.no_grad()
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

    def forward(self, pred_pose, target_pose):
        # five landmarks
        predict = self.dof2mat(pred_pose)
        label = self.dof2mat(target_pose)
        corner1 = torch.tensor([-114 / 2.0, -114 / 2.0, 0, 1], dtype=torch.float32, device=pred_pose.device).unsqueeze(
            1)
        corner2 = torch.tensor([-114 / 2.0, 114 / 2.0, 0, 1], dtype=torch.float32, device=pred_pose.device).unsqueeze(1)
        corner3 = torch.tensor([114 / 2.0, -114 / 2.0, 0, 1], dtype=torch.float32, device=pred_pose.device).unsqueeze(1)
        corner4 = torch.tensor([114 / 2.0, 114 / 2.0, 0, 1], dtype=torch.float32, device=pred_pose.device).unsqueeze(1)
        center = torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=pred_pose.device).unsqueeze(1)
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
        corner1_error = torch.norm(corner1_pred - corner1_true, dim=1)
        corner2_error = torch.norm(corner2_pred - corner2_true, dim=1)
        corner3_error = torch.norm(corner3_pred - corner3_true, dim=1)
        corner4_error = torch.norm(corner4_pred - corner4_true, dim=1)
        center_error = torch.norm(center_pred - center_true, dim=1)

        loss = torch.sum(corner1_error
                         + corner2_error
                         + corner3_error
                         + corner4_error
                         + center_error) / (5.0 * center_error.shape[0])

        return loss


class FocalLoss(_Loss):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            # print("fcls input.size", input.size(), target.size())
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)
        # print("fcls reshape input.size", input.size(), target.size())

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class LNCCLoss(torch.nn.Module):
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        super(LNCCLoss, self).__init__()
        self.win = win

    def forward(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)


class NCCLoss(nn.Module):
    def __init__(self):
        super(NCCLoss, self).__init__()

    def forward(self, y_true, y_pred):
        return -normalized_cross_correlation(y_true, y_pred)


class Grad(nn.Module):
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        super(Grad, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx

        d = torch.mean(dx) + torch.mean(dy)
        grad = d / 2.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad


class FlowLoss(nn.Module):
    def __init__(self, full_size, vol_shape=(4, 16, 16), slice_size=(16,16)):
        super(FlowLoss, self).__init__()
        self.vol_shape = vol_shape
        self.full_size = full_size
        slice_size = [1] + list(slice_size)
        vectors = [torch.linspace(-0.5 * (s - 1), 0.5 * (s - 1), steps=s) for s in slice_size]
        grids = torch.meshgrid(vectors)  # meshgrid（0,0,0,0,1,1,1,1)(0,0,1,1,0,0,1,1)(0,1,0,1,0,1,0,1),三个维度输入有顺序的
        grid = torch.stack([grids[2],grids[1],grids[0], torch.ones_like(grids[0])], dim=0)
        grid = grid.view(4, -1).type(torch.FloatTensor)

        self.register_buffer('grid', grid)
        self.sml1 = nn.SmoothL1Loss()
        self.grad = Grad(penalty='l2')

    @torch.no_grad()
    def norm_Translate(self, dof_trans):
        ntrans = torch.zeros_like(dof_trans, dtype=dof_trans.dtype, device=dof_trans.device)
        dim = len(self.vol_shape)
        for i in range(dim):
            ntrans[:,i] = dof_trans[:,i]*(self.vol_shape[dim-1-i]-1)/(self.full_size[dim-1-i]-1)
        ntrans[:, 3:] = dof_trans[:, 3:]
        return ntrans

    @torch.no_grad()
    def dof2mat(self, input_dof):
        rad = tgm.deg2rad(input_dof[:, 3:])

        ai, aj, ak = rad[:, 0], rad[:, 1], rad[:, 2]
        si, sj, sk = torch.sin(ai), torch.sin(aj), torch.sin(ak)
        ci, cj, ck = torch.cos(ai), torch.cos(aj), torch.cos(ak)
        cc, cs = ci * ck, ci * sk
        sc, ss = si * ck, si * sk
        M = torch.eye(4, dtype=input_dof.dtype, device=input_dof.device).repeat(input_dof.shape[0], 1, 1)
        I = torch.eye(4, dtype=input_dof.dtype, device=input_dof.device).repeat(input_dof.shape[0], 1, 1)
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

        return M - I

    def forward(self, pred_flow, input_dof):
        input_dof = self.norm_Translate(input_dof)
        mat = self.dof2mat(input_dof)
        flow_gt = mat @ self.grid
        flow_gt = flow_gt[:,:3].view(*pred_flow.shape)
        flow_sml1 = self.sml1(flow_gt, pred_flow)
        grad_gt = self.grad(flow_gt)
        grad_pred = self.grad(pred_flow)
        grad_sml1 = self.sml1(grad_gt, grad_pred)

        return flow_sml1 + grad_sml1
