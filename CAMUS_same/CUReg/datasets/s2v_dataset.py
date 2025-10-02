import os
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
import tools
import numpy as np
from os import path
from PIL import Image


# my dataset
class MyDataset(Dataset):

    def __init__(self, mode, root_dir, mvsIpt = True, normalize_dof = False):
        self.root = root_dir
        self.mode = mode
        self.phase = root_dir.split('/')[-1]
        self.normalize_dof = normalize_dof
        self.mvsIpt = mvsIpt

        # dof_stats = np.loadtxt('/home/jun/Desktop/project/slice2volume/dataset/0710/transforms.txt')
        # dof_means = np.mean(dof_stats, axis=0)
        # dof_std = np.std(dof_stats, axis=0)
        self.dof_means = [-0.94144243, -0.51833163, -0.24158226, -0.37933836,  0.08427333, -0.01649231]
        self.dof_std = [11.07899716, 11.08303632, 12.56951074,  5.77583911,  5.68897645, 5.8570513]
        self.slicelist = np.array([0,1,2,3])
        if mode == 'train':
            self.path = os.path.join(self.root, 'train')
        else:
            self.path = os.path.join(self.root, 'test')
        self.number_volume = (len(os.listdir(self.path))) // 32
        if mode == 'train':
            self.train_list = [i for i in range(0, self.number_volume)]
            np.random.shuffle(self.train_list)
            # print(len(self.train_list))
        else:
            self.test_vol_list = [i for i in range(0, self.number_volume)]
            self.test_slice_list = [0,1,2,3]
            self.test_list = []
            for vol in self.test_vol_list:
                for slice_id in self.test_slice_list:
                    slice_idx = [vol, slice_id]
                    self.test_list.append(slice_idx)
            # print(self.test_list)
        # print('{0} dataset: {1}'.format(self.mode, self.number_volume))

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_list)
        else:
            return len(self.test_list)
        
    def __getitem__(self, idx):

        # the directory of files
        if self.mode == 'train':
            case_id = self.train_list[idx]
            # print('self.phase {}'.format(self.phase))
            vol_idx = case_id
            target_sliceidx = np.random.randint(0, 4) ## random choice a slice
            remaining_elements = np.delete(self.slicelist, target_sliceidx)
        else:
            case_id = self.test_list[idx]
            vol_idx = case_id[0]
            target_sliceidx = case_id[1] ## random choice a slice
        
        volume_dir = os.path.join(self.path, '{0}_volume_{1}.mhd'.format(vol_idx,target_sliceidx))
        slice_dir = os.path.join(self.path, '{0}_slice_{1}.mhd'.format(vol_idx,target_sliceidx))
        para_dir = os.path.join(self.path, '{0}_s2v_{1}.txt'.format(vol_idx,target_sliceidx))

        relative_trans_dir = os.path.join(self.path, '{0}_s2ss_{1}.txt'.format(vol_idx,target_sliceidx))
        # slice_mask = os.path.join(self.path, '{0}_slice_gt_{1}.png'.format(case_id,target_sliceidx))

        ### slice mask for edge enhancement
        with Image.open(os.path.join(self.path, '{0}_slice_gt_{1}.png'.format(vol_idx,target_sliceidx))) as ri:
            slice_mask = torch.from_numpy((np.array(ri) / 255.0).astype(np.float32)).unsqueeze(0)

        # print("slice_mask shape: {0}".format(slice_mask.shape))

        # read files and their information
        # volume
        volume = sitk.ReadImage(volume_dir)
        volume_np = sitk.GetArrayFromImage(volume)  # ZYX

        # slice
        slice = sitk.ReadImage(slice_dir)
        slice_np = sitk.GetArrayFromImage(slice).squeeze(0)  # Y by X
        # remain slices
        if self.mode == 'train':
            slice_1_np = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.path, '{0}_slice_{1}.mhd'.format(vol_idx,remaining_elements[0])))).squeeze(0)
            slice_2_np = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.path, '{0}_slice_{1}.mhd'.format(vol_idx,remaining_elements[1])))).squeeze(0)
            slice_3_np = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.path, '{0}_slice_{1}.mhd'.format(vol_idx,remaining_elements[2])))).squeeze(0)
        

        # relative translation parameters
        relative_trans = torch.from_numpy(np.loadtxt(relative_trans_dir).astype(np.float32)) #[1,3]

        # transformation parameters (translation + sxyz-degree)
        self.para = np.loadtxt(para_dir)
        mat_np = tools.dof2mat_np(self.para.copy())

        # mat to 6DOF (translation + sxyz(degree))
        volume_np = np.expand_dims(volume_np, 0)
        volume_np = np.expand_dims(volume_np, 0)
        vol_tensor = torch.from_numpy(volume_np.astype(np.float32))
        vol_tensor = vol_tensor.float()
        vol_tensor = vol_tensor.squeeze(0)

        slice_np = np.expand_dims(slice_np, axis=0)
        slice_tensor = torch.from_numpy(slice_np.astype(np.float32))
        slice_tensor = slice_tensor.unsqueeze(0)
        ### remain slices
        if self.mode == 'train':
            slice_1_tensor = torch.from_numpy(np.expand_dims(slice_1_np, axis=0).astype(np.float32)).unsqueeze(0)
            slice_2_tensor = torch.from_numpy(np.expand_dims(slice_2_np, axis=0).astype(np.float32)).unsqueeze(0)
            slice_3_tensor = torch.from_numpy(np.expand_dims(slice_3_np, axis=0).astype(np.float32)).unsqueeze(0)

        ### concat slices
        if self.mode == 'train':
            slice_tensor = torch.concat((slice_tensor,slice_1_tensor,slice_2_tensor,slice_3_tensor), dim=0) # [4,1,128,128]



        mat_tensor = torch.tensor(mat_np)
        if self.normalize_dof:
            dof_tensor = torch.from_numpy((self.para - self.dof_means) / self.dof_std)
        else:
            dof_tensor = torch.from_numpy(self.para.astype(np.float32))
            # print('no norm')

        return vol_idx, vol_tensor, slice_tensor, mat_tensor, dof_tensor, relative_trans, slice_mask
    

class FreehandUS4D(Dataset):

    def __init__(self, root_dir, transform=None):
        """
        """
        samples = np.loadtxt(root_dir)
        self.samples = samples
        self.transform = transform
        self.phase = root_dir.split('/')[-1].split('.')[0]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        # the directory of files
        case_id = str(int(self.samples[idx]))
        print('self.phase {}'.format(self.phase))
        volume_dir = path.join(dataset_dir, self.phase, '{}_volume.mhd'.format(case_id))
        slice_dir = path.join(dataset_dir, self.phase, '{}_slice.mhd'.format(case_id))
        para_dir = path.join(dataset_dir, self.phase, '{}.txt'.format(case_id))

        # read files and their information
        """ Load reconstructed .mhd file into sitk img """
        self.volume = sitk.ReadImage(volume_dir)
        spacing_volume_raw = self.volume.GetSpacing()
        size_volume = self.volume.GetSize()   # XYZ
        direction_volume = self.volume.GetDirection()
        orgin_volume = self.volume.GetOrigin()

        # slice
        self.slice = sitk.ReadImage(slice_dir)
        slice_np = sitk.GetArrayFromImage(self.slice).squeeze(0)    # Y by X
        spacing_slice = self.slice.GetSpacing()
        # size_slice = self.slice.GetSize()  # XYZ
        direction_slice = self.slice.GetDirection()
        origin_slice = self.slice.GetOrigin()

        # resampling para (translation + sxyz-degree)
        self.para = np.loadtxt(para_dir)
        # slice index frame (left upper corner) to volume physical frame, including the pixelspacing in slice
        slice_mat = tools.para2Transformation(direction_slice, origin_slice, spacing_slice, self.para)

        # crop the slice
        clip_y, clip_x, clip_h, clip_w = 0, 0, slice_np.shape[0], slice_np.shape[1]
        clip_info = [clip_y, clip_x, clip_h, clip_w]
        # it works when the pixelspacing is not equal between of slice and volume and slice need to clip
        # the returned slice is with the same pixelspacing as the volume
        slice_np = tools.processFrame(us_spacing=spacing_volume_raw,
                                       frame_np=slice_np, frame_mat=slice_mat, clip_info=clip_info)

        # Generate slice_mat_gt
        # convert the slice_mat to the transformation from central physical frame of slice to the central physical
        # frame of volume
        self.volume.SetSpacing((1, 1, 1))

        self.slice_mat_gt = tools.matSitk2Stn(input_mat=slice_mat.copy(),
                                  clip_size=(clip_y, clip_x),
                                  raw_spacing=spacing_volume_raw, frame_shape=slice_np.shape,
                                  img_size=self.volume.GetSize(),
                                  img_spacing=self.volume.GetSpacing(),
                                  img_origin=self.volume.GetOrigin())

        """ Initialize a subvolume using init_mat """
        cropvol_size = (128, 128, 32)
        cropslice_size = (128, 128)
        # central crop, the dimension of  vol_crop is ZYX
        vol_crop = tools.sampleSubvol(sitk_img=self.volume, init_mat=np.identity(4),
                                      crop_size=cropvol_size)
        # central crop
        slice_crop = tools.frameCrop(input_np=slice_np, crop_size=cropslice_size)

        # mat to 6DOF (translation + sxyz(degree))
        diff_mat = self.slice_mat_gt.copy()
        diff_dof = tools.mat2dof_np(input_mat=diff_mat)
        diff_dof_copy = diff_dof.copy()

        vol_crop = np.expand_dims(vol_crop, 0)
        vol_crop = np.expand_dims(vol_crop, 0)
        vol_tensor = torch.from_numpy(vol_crop)
        vol_tensor = vol_tensor.float()
        vol_tensor = vol_tensor.squeeze(0)

        mat_tensor = torch.tensor(diff_mat)
        dof_tensor = torch.tensor(diff_dof)


        slice_crop = np.expand_dims(slice_crop, axis=0)
        slice_tensor = torch.from_numpy(slice_crop)
        slice_tensor = slice_tensor.unsqueeze(0)


        if normalize_dof:
            dof_tensor = torch.from_numpy(diff_dof_copy[:6])
            # print('no norm')
        else:
            dof_tensor = torch.from_numpy((diff_dof_copy[:6]-dof_means)/dof_std)

        return case_id, vol_tensor, slice_tensor, mat_tensor, dof_tensor
    


if __name__ == "__main__":
    # main()
    data_root = '/home/jun/Desktop/project/slice2volume/dataset/CAMUS-0129'
    dataset = MyDataset("test", data_root)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=12)
    # invisible = []
    # loss_rate=[]
    for i, data in enumerate(dataloader):
        # print("val: {0}/{1}".format(i,900),end='\r')
        vol_idx, vol_tensor, frame_tensor, mat_tensor, dof_tensor, relative_trans, slice_mask = data

        # print("vol_idx shape: {0}".format(vol_idx.shape))
        print("vol_tensor shape: {0}".format(vol_tensor.shape))
        print("vol_tensor: {0}".format(vol_tensor))
        print("frame_tensor shape: {0}".format(frame_tensor.shape))
        print("frame_tensor: {0}".format(frame_tensor))
        # print("mat_tensor shape: {0}".format(mat_tensor.shape))
        # print("dof_tensor shape: {0}".format(dof_tensor.shape))
        # print("relative_trans shape: {0}".format(relative_trans.shape))
        # print("slice_mask shape: {0}".format(slice_mask.shape))
        # print("slice_mask shape: {0}".format(slice_mask))
        
        if i > 10:
            break