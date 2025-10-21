import numpy as np
import os
import glob
import SimpleITK as sitk


def get_origin(segmented_files_path):
    lung_files = glob.glob(os.path.join(segmented_files_path, 'lung*.nii.gz'))
    ref_file = sitk.GetArrayFromImage(sitk.ReadImage(lung_files[0]))

    lung_volume = np.zeros(ref_file.shape)
    for file in lung_files:
        lung_volume += sitk.GetArrayFromImage(sitk.ReadImage(file))

    metal_origin_set = np.argwhere(lung_volume > 0)
    centroid = np.round(metal_origin_set.mean(axis=0))
    origin = np.array(ref_file.shape)/2
    origin_offset = centroid - origin
    return origin_offset

