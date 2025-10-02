import math
import torchgeometry as tgm
import torch.nn.functional as F
import torch
from torch.distributions.normal import Normal
from torch import nn


class ConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True,
                 negative_slope=0.1, norm='bn'):
        super(ConvBlock2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        else:
            self.norm = nn.Identity()
        self.leakyrelu = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.leakyrelu(x)
        return x


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True,
                 negative_slope=0.1, norm='bn'):
        super(ConvBlock3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        if norm == 'bn':
            self.norm = nn.BatchNorm3d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm3d(out_channels)
        else:
            self.norm = nn.Identity()
        self.leakyrelu = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.leakyrelu(x)
        return x


class ProjectLN3D(nn.Module):
    def __init__(self, in_channels, dim=6, norm=nn.LayerNorm):
        super().__init__()
        self.norm = norm(dim)
        self.proj = nn.Linear(in_channels, dim)
        self.proj.weight = nn.Parameter(Normal(0, 1e-5).sample(self.proj.weight.shape))
        self.proj.bias = nn.Parameter(torch.zeros(self.proj.bias.shape))

    def forward(self, feat):
        feat = feat.permute(0, 2, 3, 4, 1)
        feat = self.norm(self.proj(feat))
        return feat


class ProjectLN2D(nn.Module):
    def __init__(self, in_channels, dim=6, norm=nn.LayerNorm):
        super().__init__()
        self.norm = norm(dim)
        self.proj = nn.Linear(in_channels, dim)
        self.proj.weight = nn.Parameter(Normal(0, 1e-5).sample(self.proj.weight.shape))
        self.proj.bias = nn.Parameter(torch.zeros(self.proj.bias.shape))

    def forward(self, feat):
        feat = feat.permute(0, 2, 3, 1)
        feat = self.norm(self.proj(feat))
        return feat


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

class Encoder2D(nn.Module):
    def __init__(self, in_channel, first_channel=8):
        super(Encoder2D, self).__init__()

        c = first_channel
        self.layer1 = nn.Sequential(
            ConvBlock2D(in_channel, c),
            ConvBlock2D(c, c),
        )

        self.layer2 = nn.Sequential(
            nn.AvgPool2d(2),
            ConvBlock2D(c, 2 * c),
            ConvBlock2D(2 * c, 2 * c),
        )

        self.layer3 = nn.Sequential(
            nn.AvgPool2d(2),
            ConvBlock2D(2 * c, 4 * c),
            ConvBlock2D(4 * c, 4 * c),
        )

        self.layer4 = nn.Sequential(
            nn.AvgPool2d(2),
            ConvBlock2D(4 * c, 8 * c),
            ConvBlock2D(8 * c, 8 * c),
        )
        #

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        return out4  # , out2, out3, out4


class Encoder3D(nn.Module):
    def __init__(self, in_channel, first_channel=8):
        super(Encoder3D, self).__init__()

        c = first_channel
        self.layer1 = nn.Sequential(
            ConvBlock3D(in_channel, c),
            ConvBlock3D(c, c),
        )

        self.layer2 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvBlock3D(c, 2 * c),
            ConvBlock3D(2 * c, 2 * c),
        )

        self.layer3 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvBlock3D(2 * c, 4 * c),
            ConvBlock3D(4 * c, 4 * c),
        )

        self.layer4 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvBlock3D(4 * c, 8 * c),
            ConvBlock3D(8 * c, 8 * c),
        )

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        return out4  # , out2, out3, out4


class CDFE(nn.Module):
    def __init__(self, vol_inshape, slice_inshape, in_channels_2d, in_channels_3d, dim):
        super(CDFE, self).__init__()
        '''
        vol_inshape: shape of input volume(T, H, W)
        slice_inshape: shape of input slice(H, W)
        '''
        vectors = [torch.linspace(-0.5 * (s - 1), 0.5 * (s - 1), steps=s) for s in vol_inshape]
        grids = torch.meshgrid(vectors)
        G_vol = torch.stack(grids, dim=-1).view(-1, 3).type(torch.FloatTensor).contiguous()  #(T,H,W,3)->(THW, 3)
        self.register_buffer('G_vol', G_vol)

        slice_size = [1] + list(slice_inshape)
        vectors = [torch.linspace(-0.5 * (s - 1), 0.5 * (s - 1), steps=s) for s in slice_size]
        slice_grids = torch.meshgrid(vectors)
        G_slice = torch.stack(slice_grids, dim=1).type(torch.FloatTensor).contiguous()
        self.register_buffer('G_slice', G_slice)

        self.proj3d = ProjectLN3D(in_channels_3d, dim)
        self.proj2d = ProjectLN2D(in_channels_2d, dim)

    def forward(self, vol, slice):
        '''
        vol: input volume(B, C, T, H, W)
        vol: input slice(B, C, H, W)
        '''
        B, C, T, H, W = vol.shape
        Vs, Ss = T * H * W, H * W

        Q = self.proj2d(slice).view(B, Ss, -1)  # B, H, W, d -> B, HW, d
        K = self.proj3d(vol).view(B, Vs, -1)  # B, T, H, W, d -> B, THW, d

        Attn = Q @ K.transpose(1, 2)
        Attn = Attn.softmax(dim=-1)  # (B, H*W, T*H*W)

        G_pred = Attn @ self.G_vol  # (B, H*W, 3)
        flow = G_pred.transpose(1, 2).view(B, 3, H, W) - self.G_slice

        return flow

class TransNet(nn.Module):
    def __init__(self, full_vol_size, deep_vol_size, input_channels=3, slope=0.01):
        super(TransNet, self).__init__()
        self.full_vol_size = full_vol_size
        self.deep_vol_size = deep_vol_size
        self.vol_shape = deep_vol_size
        # Feature extraction with convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AvgPool2d(2, 2)  # Down-sample by 2
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.relu = nn.LeakyReLU(negative_slope=slope)
        # Fully connected layers for regression
        # Global feature aggregation
        self.fc1 = nn.Linear(128 * (deep_vol_size[1] // 8) * (deep_vol_size[2] // 8),
                             256)  # Adjust for pooled spatial size
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 3)  # Output 3 rotation parameters (roll, pitch, yaw)
        self.fc3.weight = nn.Parameter(Normal(0, 1e-5).sample(self.fc3.weight.shape))
        self.fc3.bias = nn.Parameter(torch.zeros(self.fc3.bias.shape))

    def norm_Translate(self, dof_trans):
        ntrans = torch.zeros_like(dof_trans, dtype=dof_trans.dtype, device=dof_trans.device)
        for i in range(len(self.full_vol_size)):
            ntrans[:, i] = dof_trans[:, i] * (self.full_vol_size[2-i] - 1) / (self.deep_vol_size[2-i] - 1)  # dof(_,_,fr) revsize(_,_,fr)
        return ntrans

    def forward(self, x):
        # flow_avg = self.global_pool(x)

        # Feature extraction
        x = self.relu(self.conv1(x))  # (B, 32, H, W)
        x = self.pool(x)  # (B, 32, H/2, W/2)
        x = self.relu(self.conv2(x))  # (B, 64, H/2, W/2)
        x = self.pool(x)  # (B, 64, H/4, W/4)
        x = self.relu(self.conv3(x))  # (B, 128, H/4, W/4)
        x = self.pool(x)  # (B, 128, H/8, W/8)

        # Flatten
        x = x.reshape(x.size(0), -1)  # (B, 128 * (H/8) * (W/8))

        # Fully connected layers
        x = self.relu(self.fc1(x))  # (B, 256)
        x = self.relu(self.fc2(x))  # (B, 128)
        dof_trans = self.fc3(x)  # (B, 3)
        return dof_trans.unsqueeze(-1).unsqueeze(-1), self.norm_Translate(dof_trans)


class RotationNet(nn.Module):
    def __init__(self, vol_shape, input_channels=3, slope=0.01):
        super(RotationNet, self).__init__()

        self.vol_shape = vol_shape
        self.R = math.sqrt((vol_shape[1] // 2) ** 2 + (vol_shape[2] // 2) ** 2)
        # Feature extraction with convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # Down-sample by 2

        self.relu = nn.LeakyReLU(negative_slope=slope)
        # Fully connected layers for regression
        # Global feature aggregation
        self.fc1 = nn.Linear(128 * (vol_shape[1] // 8) * (vol_shape[2] // 8), 256)  # Adjust for pooled spatial size
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 3)  # Output 3 rotation parameters (roll, pitch, yaw)
        self.fc3.weight = nn.Parameter(Normal(0, 1e-5).sample(self.fc3.weight.shape))
        self.fc3.bias = nn.Parameter(torch.zeros(self.fc3.bias.shape))

    def norm_rot_flow(self, flow, dof_trans):
        n_rotflow = flow - dof_trans
        return n_rotflow/self.R

    def forward(self, x, dof_trans):
        x = self.norm_rot_flow(x, dof_trans)
        # Feature extraction
        x = self.relu(self.conv1(x))  # (B, 32, H, W)
        x = self.pool(x)  # (B, 32, H/2, W/2)
        x = self.relu(self.conv2(x))  # (B, 64, H/2, W/2)
        x = self.pool(x)  # (B, 64, H/4, W/4)
        x = self.relu(self.conv3(x))  # (B, 128, H/4, W/4)
        x = self.pool(x)  # (B, 128, H/8, W/8)

        # Flatten
        x = x.reshape(x.size(0), -1)  # (B, 128 * (H/8) * (W/8))

        # Fully connected layers
        x = self.relu(self.fc1(x))  # (B, 256)
        x = self.relu(self.fc2(x))  # (B, 128)
        rotation_params = self.fc3(x)  # (B, 3)
        return rotation_params


class EUReg(nn.Module):
    def __init__(self, vol_shape, in_channel=1, first_channel=8, dim=6):
        super(EUReg, self).__init__()
        c = first_channel
        self.encoder3d = Encoder3D(in_channel=in_channel, first_channel=c)
        self.encoder2d = Encoder2D(in_channel=in_channel, first_channel=c)

        L = 4
        deep_vol_shape = [s // 2 ** (L - 1) for s in vol_shape]
        self.deep_vol_shape = deep_vol_shape

        self.CDFE = CDFE(vol_inshape=deep_vol_shape, slice_inshape=deep_vol_shape[1:],
                       in_channels_2d=2 ** (L - 1) * c, in_channels_3d=2 ** (L - 1) * c, dim=dim)

        self.transnet = TransNet(vol_shape, deep_vol_shape)
        self.rotnet = RotationNet(deep_vol_shape, input_channels=3)

    def forward(self, vol, slice):
        Fv = self.encoder3d(vol)
        Fs = self.encoder2d(slice)

        flow = self.CDFE(Fv, Fs).flip(dims=[1]) # change: (W,H,frame)

        deep_dof_trans, pred_trans = self.transnet(flow)
        pred_rot = self.rotnet(flow, deep_dof_trans)

        return torch.cat([pred_trans, pred_rot], dim=1), flow

class EUReg_Flow(nn.Module):
    def __init__(self, vol_shape, in_channel=1, first_channel=8, dim=6):
        super(EUReg_Flow, self).__init__()
        c = first_channel
        self.encoder3d = Encoder3D(in_channel=in_channel, first_channel=c)
        self.encoder2d = Encoder2D(in_channel=in_channel, first_channel=c)

        L = 4  # 4
        deep_vol_shape = [s // 2 ** (L - 1) for s in vol_shape]
        self.deep_vol_shape = deep_vol_shape

        self.CDFE = CDFE(vol_inshape=deep_vol_shape, slice_inshape=deep_vol_shape[1:],
                       in_channels_2d=2 ** (L - 1) * c, in_channels_3d=2 ** (L - 1) * c, dim=dim)

    def forward(self, vol, slice):
        Fv = self.encoder3d(vol)
        Fs = self.encoder2d(slice)
        flow = self.CDFE(Fv, Fs).flip(dims=[1])

        return flow


class EUReg_FRT(EUReg):
    def __init__(self, vol_shape, in_channel=1, first_channel=8, dim=6):
        super().__init__(vol_shape, in_channel, first_channel, dim)

        self.transformer = FrameRigidTransformer(vol_shape[1:])

    def forward(self, vol, slice):
        pred_dof, flow = super().forward(vol, slice)
        moved = self.transformer(vol, pred_dof).squeeze(2)

        return pred_dof, flow, moved

class EUReg_FRT_Test(EUReg):
    def __init__(self, vol_shape, in_channel=1, first_channel=8, dim=6):
        super().__init__(vol_shape, in_channel, first_channel, dim)

        self.transformer = FrameRigidTransformer(vol_shape[1:])

    def forward(self, vol, slice):
        pred_dof, _ = super().forward(vol, slice)
        moved = self.transformer(vol, pred_dof).squeeze(2)

        return pred_dof, moved

if __name__ == '__main__':
    model = EUReg((32, 128, 128)).cuda()
    model2 = EUReg_FRT((32, 128, 128)).cuda()
    f = torch.ones(2, 1, 128, 128).cuda()
    v = torch.ones(2, 1, 32, 128, 128).cuda()
    output = model(v, f)
    for a in output:
        print(a.shape)

    output = model2(v, f)
    for a in output:
        print(a.shape)