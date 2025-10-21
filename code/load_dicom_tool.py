import pydicom as dicom
import glob
import numpy as np
from skimage.transform import resize
import segmentation
import SimpleITK as sitk
import segmented_files2materials
import os
import matplotlib.pyplot as plt
import nibabel as nib
from tqdm import tqdm
materials = {1: "air", 2: "soft tissue", 3: "cortical bone"}

def replace_material(metal_volume_m_ori, smooth_air = False, use_thresholding_segmentation=True):

    volume = metal_volume_m_ori

    #convert hu_values to materials
    if not use_thresholding_segmentation:
        materials = conv_hu_to_materials(volume)
    else:
        materials = conv_hu_to_materials_thresholding(volume)

    return materials

def load_dicom_CT(source_path =r"./*/*/", fixed_slice_thinckness = None, new_resolution = None, truncate = None, smooth_air = False, use_thresholding_segmentation = False, file_extension = ".dcm"):
    #source_path += "*"+ file_extension
    source_path += "/*.dcm"
    files = np.array(glob.glob(source_path))
    one_slice = dicom.read_file(files[0], force=True)
    if hasattr(one_slice, "InstanceNumber"):
        sliceOrder = [dicom.read_file(curDCM, force=True).InstanceNumber for curDCM in files]
        files = files[np.argsort(sliceOrder).astype(np.int32)]
    else:
        sliceOrder = [dicom.read_file(curDCM, force=True).SliceLocation for curDCM in files]
        files = files[np.argsort(sliceOrder).astype(np.int32)]

    files = list(files)

    # Get ref file
    refDs = dicom.read_file(files[0], force=True)


    # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
    volume_size = [int(refDs.Rows), int(refDs.Columns), files.__len__()]

    # if not hasattr(refDs,"SliceThickness"):
    #     print('Volume has no attribute Slice Thickness, please provide it manually!')
    #     print('using fixed slice thickness of:', fixed_slice_thinckness)
    #     voxel_size = [float(refDs.PixelSpacing[1]), float(refDs.PixelSpacing[0]), fixed_slice_thinckness]
    # else:
    voxel_size = [float(refDs.PixelSpacing[1]), float(refDs.PixelSpacing[0]), float(refDs.SliceThickness)]

    # The array is sized based on 'PixelDims'
    volume = np.zeros(volume_size, dtype=np.float64)

    # loop through all the DICOM files
    for filenameDCM in files:
        # read the file
        ds = dicom.read_file(filenameDCM, force=True)
        # store the raw image data
        if files.index(filenameDCM) < volume.shape[2]:
            volume[:, :, files.index(filenameDCM)] = ds.pixel_array.astype(np.int32)

    #use intercept point
    if hasattr(refDs, "RescaleIntercept"):
        volume += int(refDs.RescaleIntercept)

    volume = np.moveaxis(volume, [0, 1, 2], [1, 0, 2]).copy()

    #truncate
    if truncate:
        volume = volume[truncate[0][0]:truncate[0][1],truncate[1][0]:truncate[1][1],truncate[2][0]:truncate[2][1]]

    # volume = np.flip(volume,2)
    #upsample Volume
    if new_resolution:
        volume, volume_size, voxel_size = upsample(volume, new_resolution, voxel_size)

    #convert hu_values to density

    densities = conv_hu_to_density(volume, smoothAir= smooth_air)

    #convert hu_values to materials
    if not use_thresholding_segmentation:
        materials = conv_hu_to_materials(volume)
    else:
        materials = conv_hu_to_materials_thresholding(volume)

    return volume, densities.astype(np.float32), materials, np.array(voxel_size,dtype=np.float32)
    
def load_dicom_CT_new(original_files_path = None, segmented_files_path = None, fixed_slice_thinckness = None, new_resolution = None, truncate = None, smooth_air = False, use_thresholding_segmentation = False, file_extension = ".dcm"):
    #source_path += "*"+ file_extension
    original_files_path += "/*.dcm"
    files = np.array(glob.glob(original_files_path))
    one_slice = dicom.read_file(files[0], force=True)
    if hasattr(one_slice, "InstanceNumber"):
        sliceOrder = [dicom.read_file(curDCM, force=True).InstanceNumber for curDCM in files]
        files = files[np.argsort(sliceOrder).astype(np.int32)]
    else:
        sliceOrder = [dicom.read_file(curDCM, force=True).SliceLocation for curDCM in files]
        files = files[np.argsort(sliceOrder).astype(np.int32)]

    files = list(files)

    # Get ref file
    refDs = dicom.read_file(files[0], force=True)

    # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
    volume_size = [int(refDs.Rows), int(refDs.Columns), files.__len__()]

    if not hasattr(refDs,"SliceThickness"):
        print('Volume has no attribute Slice Thickness, please provide it manually!')
        print('using fixed slice thickness of:', fixed_slice_thinckness)
        voxel_size = [float(refDs.PixelSpacing[1]), float(refDs.PixelSpacing[0]), fixed_slice_thinckness]
    else:
        voxel_size = [float(refDs.PixelSpacing[1]), float(refDs.PixelSpacing[0]), float(refDs.SliceThickness)]

    # The array is sized based on 'PixelDims'
    volume = np.zeros(volume_size, dtype=np.float64)

    # loop through all the DICOM files
    for filenameDCM in files:
        # read the file
        ds = dicom.read_file(filenameDCM, force=True)
        # store the raw image data
        if files.index(filenameDCM) < volume.shape[2]:
            volume[:, :, files.index(filenameDCM)] = ds.pixel_array.astype(np.int32)

    #use intercept point
    if hasattr(refDs, "RescaleIntercept"):
        volume += int(refDs.RescaleIntercept)

    volume = np.moveaxis(volume, [0, 1, 2], [1, 0, 2]).copy()

    #truncate
    if truncate:
        volume = volume[truncate[0][0]:truncate[0][1],truncate[1][0]:truncate[1][1],truncate[2][0]:truncate[2][1]]

    # volume = np.flip(volume,2)
    #upsample Volume
    if new_resolution:
        volume, volume_size, voxel_size = upsample(volume, new_resolution, voxel_size)

    #convert hu_values to density
    densities = conv_hu_to_density(volume, smoothAir= smooth_air)

    #convert hu_values to materials
    segmented_files = glob.glob(os.path.join(segmented_files_path, '*.nii.gz'))
    print(f"find {len(segmented_files)} segmented files")
    ref_file = sitk.GetArrayFromImage(sitk.ReadImage(segmented_files[0]))

    bone_volume = np.zeros(ref_file.shape)
    soft_tissue_volume = np.zeros(ref_file.shape)
    air_volume = np.ones(ref_file.shape)

    for segmented_file in segmented_files:
        segmented_file_array = sitk.GetArrayFromImage(sitk.ReadImage(segmented_file))
        file_name = os.path.basename(segmented_file)
        if segmented_files2materials.segmented_files2materials[file_name[:-7]] == 'soft tissue':
            soft_tissue_volume += segmented_file_array.astype(np.int8)
        if segmented_files2materials.segmented_files2materials[file_name[:-7]] == 'bone':
            bone_volume += segmented_file_array.astype(np.int8)

    air_volume = air_volume - soft_tissue_volume - bone_volume

    soft_tissue_volume = np.moveaxis(soft_tissue_volume, [0, 1, 2], [2, 1, 0]).copy()
    soft_tissue_volume = np.flip(soft_tissue_volume, axis=1)
    soft_tissue_volume = np.flip(soft_tissue_volume, axis=2)
    bone_volume = np.moveaxis(bone_volume, [0, 1, 2], [2, 1, 0]).copy()
    bone_volume = np.flip(bone_volume, axis=1)
    bone_volume = np.flip(bone_volume, axis=2)
    air_volume = np.moveaxis(air_volume, [0, 1, 2], [2, 1, 0]).copy()
    air_volume = np.flip(air_volume, axis=1)
    air_volume = np.flip(air_volume, axis=2)


    materials = {"soft tissue": soft_tissue_volume.astype(bool), "bone": bone_volume.astype(bool),
                "air": air_volume.astype(bool)}
    # materials['soft tissue'] = 0 * materials['soft tissue']
    # materials['air'] = 0 * materials['air']
    # plt.imshow(segmented_volume[:,:,segmented_volume.shape[2]//2])
    # plt.show()

    # plt.imshow(segmented_volume[:,segmented_volume.shape[1]//2,:])
    # plt.show()
    # plt.imshow(segmented_volume[segmented_volume.shape[0]//2,:,:])
    # plt.show()
    return volume, densities.astype(np.float32), materials, np.array(voxel_size,dtype=np.float32)

def load_dicom_metal(source_path =r"./*/*/", metal_voxel_size=[0.1,0.1,0.1],metal_orientation = np.arange(3), sortBy ="SliceLocation", fixed_slice_thinkness = None, new_resolution = [100,100,100], truncate = None, smooth_air = False, use_thresholding_segmentation = False, flip=False, density_metal=None):
    ##Metal Volume
    files = np.array(glob.glob(source_path))
    one_slice_body = dicom.read_file(files[0],force=True)
    if hasattr(one_slice_body, "InstanceNumber"):
        sliceOrder = [dicom.read_file(curDCM,force=True).InstanceNumber for curDCM in files]
        files = files[np.argsort(sliceOrder).astype(np.int8)]
    else:
        sliceOrder = [dicom.read_file(curDCM,force=True).SliceLocation for curDCM in files]
        files = files[np.argsort(sliceOrder).astype(np.int8)]

        files = list(files)

    # Get ref file
    refDs_body = dicom.read_file(files[0],force=True)

    volume_size = [int(refDs_body.Rows), int(refDs_body.Columns), int(refDs_body.NumberOfFrames)] # The last number needs to be changed   
    voxel_spacing = float(refDs_body.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing[0])
    # voxel_size = [voxel_spacing, voxel_spacing, voxel_spacing] # change size of rfo
    # voxel_size = [0.1, 0.1, 0.1]
    metal_voxel_size = [0.2,0.2,0.2]
    voxel_size = metal_voxel_size

    # The array is sized based on 'PixelDims'
    volume = np.zeros(volume_size, dtype=np.float32)

    # loop through all the DICOM files
    ds = dicom.read_file(files[0],force=True)
    for index in range(int(refDs_body.NumberOfFrames)):
        # read the file
        # store the raw image data
        volume[:, :, index] = ds.pixel_array[index].astype(np.int16)

    # volume = np.moveaxis(volume, [0, 1, 2], [1, 0, 2]).copy()

    volume=np.transpose(volume, metal_orientation.tolist())  #change orientation of rfo
    #truncate
    if truncate:
        volume = volume[truncate[0][0]:truncate[0][1],truncate[1][0]:truncate[1][1],truncate[2][0]:truncate[2][1]]

    #upsample Volume
    # if new_resolution:
    #     volume, volume_size, voxel_size = upsample(volume, new_resolution, voxel_size)
    
    #convert hu_values to density
    #apply density
    # volume=np.zeros(volume.shape)

    densities = volume * density_metal

    #flip densities
    if flip:
        densities = np.flip(densities, 0)

    #convert hu_values to materials
    materials = {}
    materials[os.path.basename(source_path)] = volume > 0
    #source_path[-15:-4]
    return densities.astype(np.float32), materials, np.array(voxel_size,dtype=np.float32)


def load_nifti_CT(source_path, segmented_files_path=None,fixed_slice_thickness=None, new_resolution=None, truncate=None, smooth_air=False,
                  use_thresholding_segmentation=False):

    # Load the NIfTI file
    nifti = nib.load(source_path)
    volume = nifti.get_fdata()  # Load the 3D image data as a NumPy array
    affine = nifti.affine  # Transformation matrix
    header = nifti.header  # NIfTI header

    # Extract voxel size (resolution)
    voxel_size = header.get_zooms()[:3]  # (x, y, z) voxel dimensions in mm

    # Ensure the volume is in the correct format (float64 for processing)
    volume = volume.astype(np.float64)

    # Use Rescale Intercept if available (NIfTI files usually store data directly in HU)
    # if header['scl_slope'] is not None and header['scl_inter'] is not None:
    #     volume = volume * header['scl_slope'] + header['scl_inter']

    # Move axes if needed (match to [Z, Y, X] order as in DICOM)
    # volume = np.moveaxis(volume, [0, 1, 2], [1, 0, 2]).copy()

    # Truncate volume if specified
    if truncate:
        volume = volume[
                 truncate[0][0]:truncate[0][1],
                 truncate[1][0]:truncate[1][1],
                 truncate[2][0]:truncate[2][1]
                 ]

    # Upsample Volume if new resolution is specified
    if new_resolution:
        volume, _, voxel_size = upsample(volume, new_resolution, voxel_size)

    # Convert HU values to densities
    densities = conv_hu_to_density(volume, smoothAir=smooth_air)


    segmented_files = glob.glob(os.path.join(segmented_files_path, '*.nii.gz'))
    print(f"find {len(segmented_files)} segmented files")
    ref_file = nib.load(segmented_files[0]).get_fdata()

    bone_volume = np.zeros(ref_file.shape)
    soft_tissue_volume = np.zeros(ref_file.shape)
    air_volume = np.ones(ref_file.shape)

    for segmented_file in tqdm(segmented_files, desc="loading segmentation", leave=False):
        segmented_file_array = nib.load(segmented_file).get_fdata()
        file_name = os.path.basename(segmented_file)
        if segmented_files2materials.segmented_files2materials[file_name[:-7]] == 'soft tissue':
            soft_tissue_volume += segmented_file_array.astype(np.int8)
        if segmented_files2materials.segmented_files2materials[file_name[:-7]] == 'bone':
            bone_volume += segmented_file_array.astype(np.int8)

    air_volume = air_volume - soft_tissue_volume - bone_volume

    # Depends on your coordinate system, you may need some of the following operations
    # soft_tissue_volume = np.moveaxis(soft_tissue_volume, [0, 1, 2], [2, 1, 0]).copy()
    # bone_volume = np.moveaxis(bone_volume, [0, 1, 2], [2, 1, 0]).copy()
    # air_volume = np.moveaxis(air_volume, [0, 1, 2], [2, 1, 0]).copy()
    # soft_tissue_volume = np.flip(soft_tissue_volume, axis=0)
    # soft_tissue_volume = np.flip(soft_tissue_volume, axis=1)
    # soft_tissue_volume = np.flip(soft_tissue_volume, axis=2)
    # bone_volume = np.flip(bone_volume, axis=0)
    # bone_volume = np.flip(bone_volume, axis=1)
    # bone_volume = np.flip(bone_volume, axis=2)
    # air_volume = np.flip(air_volume, axis=0)
    # air_volume = np.flip(air_volume, axis=1)
    # air_volume = np.flip(air_volume, axis=2)

    materials = {"soft tissue": soft_tissue_volume.astype(bool), "bone": bone_volume.astype(bool),
                "air": air_volume.astype(bool)}

    # if not use_thresholding_segmentation:
    #     materials = conv_hu_to_materials(volume)
    # else:
    #     materials = conv_hu_to_materials_thresholding(volume)

    return volume, densities.astype(np.float32), materials, np.array(voxel_size, dtype=np.float32)


def upsample(volume,newResolution,voxelSize):
    upsampled_voxel_size = list(np.array(voxelSize) * np.array(volume.shape) / newResolution)
    upsampled_volume = resize(volume,newResolution,order = 1,cval=-1000)
    return upsampled_volume, upsampled_voxel_size, upsampled_voxel_size

def conv_hu_to_density(hu_values, smoothAir = False):
    #Use two linear interpolations from data: (HU,g/cm^3)
    # -1000 0.00121000000000000
    #-98    0.930000000000000
    #-97    0.930486000000000
    #14 1.03000000000000
    #23 1.03100000000000
    #100    1.11990000000000
    #101    1.07620000000000
    #1600   1.96420000000000
    #3000   2.80000000000000
    # use fit1 for lower HU: density = 0.001029*HU + 1.030 (fit to first 4)
    # use fit2 for upper HU: density = 0.0005886*HU + 1.03 (fit to last 5)

    #set air densities
    if smoothAir:
        hu_values[hu_values <= -900] = -1000
    # hu_values[hu_values > 600] = 5000;
    densities = np.maximum(np.minimum(0.001029 * hu_values + 1.030, 0.0005886 * hu_values + 1.03), 0);
    return densities

def conv_hu_to_materials_thresholding(hu_values):
# ranges taken from schneider and Buzug CT
#     materials = np.zeros(hu_values.shape,dtype=np.int32)
#
    # materials[hu_values <= -800] = 1
    #
    # # Lung
    # mask = (-800 < hu_values) * (hu_values <= -200)
    # materials[mask] = 9;
    #
    # # Fat
    # mask = (-200 < hu_values) * (hu_values <= -75)
    # materials[mask] = 6;
    #
    # # Connective Tissue
    # mask = (-75 < hu_values) * (hu_values <= -5)
    # materials[mask] = 8;
    #
    # # Water
    # mask = (-5 < hu_values) * (hu_values <= 5)
    # materials[mask] = 15;
    #
    # # Soft Tissue
    # mask = (5 < hu_values) * (hu_values <= 35)
    # materials[mask] = 3;
    #
    # # Muscle
    # mask = (35 < hu_values) * (hu_values <= 50)
    # materials[mask] = 2;
    #
    # # Blood
    # mask = (50 < hu_values) * (hu_values <= 60)
    # materials[mask] = 7;
    #
    # # Liver
    # mask = (60 < hu_values) * (hu_values <= 100)
    # materials[mask] = 13;
    #
    # # Bone Marrow
    # mask = (100 < hu_values) * (hu_values <= 400)
    # materials[mask] = 12;
    #
    # # Bone
    # mask = (400 < hu_values) * (hu_values <= 3000)
    # materials[mask] = 4;
    #
    # # Titanium
    # mask = 3000 < hu_values
    # materials[mask] = 5;

    # # Air
    # materials[hu_values <= -800] = 1
    #
    # # Soft Tissue
    # mask = (-800 < hu_values) * (hu_values <= 500)
    # materials[mask] = 2;
    #
    # # Bone
    # mask = 500 < hu_values
    # materials[mask] = 3;

    materials = {}
    # Air
    materials["air"] = hu_values <= -800

    # Soft Tissue
    materials["soft tissue"] = (-800 < hu_values) * (hu_values <= 300)

    # Bone
    materials["bone"] = (300 < hu_values)

    # plt.imshow(materials["bone"][176,:,:])
    # plt.show()

    return materials

def conv_hu_to_materials(hu_values):
    segmentation_network = segmentation.SegmentationNet()
    materials = segmentation_network.segment(hu_values)


    return materials
