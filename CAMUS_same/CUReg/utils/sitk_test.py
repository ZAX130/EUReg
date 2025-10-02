import SimpleITK as sitk
import numpy as np

destVol = sitk.Image(20, 20, 20, sitk.sitkUInt8)
destVol.SetSpacing([1, 1, 1])
destVol.SetOrigin([10, 0, 20])

print('before_TransformIndexToPhysicalPoint {}'.format(destVol.TransformIndexToPhysicalPoint([1,1,1])))
destVol.SetSpacing([2, 2, 2])
print('after_TransformIndexToPhysicalPoint {}'.format(destVol.TransformIndexToPhysicalPoint([1,1,1])))