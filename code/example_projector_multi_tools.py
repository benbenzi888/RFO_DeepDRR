import glob

import projector
import projection_matrix
from load_dicom import load_dicom
from load_dicom_tool import load_dicom_metal, load_dicom_CT, load_dicom_CT_new, load_nifti_CT, replace_material
from downsample_tool import downsample_tool
from analytic_generators import add_noise
import mass_attenuation_gpu as mass_attenuation
import spectrum_generator
import add_scatter
from utils import image_saver, Camera, param_saver
import os
import matplotlib.pyplot as plt
import numpy as np
import materials
import draw_box
import get_origin

def add_metal_into_foward_projections(forward_projections,proj_mats,ori_CT_volume, CT_volume, CT_voxel_size,CT_materials,origin,camera,metal_path,origin_metal):
    metal_volume, metal_materials, metal_voxel_size = load_dicom_metal(metal_path, fixed_slice_thinkness=1.0,
                                                                       sortBy="InstanceNumber",
                                                                       use_thresholding_segmentation=True,density_metal=materials.density_dict[os.path.basename(metal_path)])
    #replace metal material with CT materials
    metal_materials_n = replace_material(metal_volume, smooth_air=False, use_thresholding_segmentation=True)
    #upsamle metal volume to CT volume
    metal_volume_m, metal_volume_m_ori, metal_materials_m = downsample_tool(ori_CT_volume, CT_volume, CT_voxel_size,
                                                                            metal_volume, metal_voxel_size,
                                                                            CT_materials, metal_materials_n, origin,
                                                                            origin_metal)
    ##forward metal project densities of metal material
    metal_projections = projector.generate_projections(proj_mats, metal_volume, metal_materials, origin_metal,
                                                       metal_voxel_size, camera.sensor_width, camera.sensor_height,
                                                       mode="linear", max_blockind=200, threads=8)
    #forward metal project densities of CT materials
    metal_projections_m = projector.generate_projections(proj_mats, metal_volume_m, metal_materials_m, origin_metal,
                                                         metal_voxel_size, camera.sensor_width, camera.sensor_height,
                                                         mode="linear", max_blockind=200, threads=8)
    #subtract in projection domain
    for mat in CT_materials:
        forward_projections[mat] = forward_projections[mat] - metal_projections_m[mat]
    ##add back metal projection
    forward_projections.update(metal_projections)

    return forward_projections, metal_voxel_size



def generate_projections_on_sphere(CT_volume_path,metal1_path,metal2_path,segmented_files_path,save_path,min_theta,max_theta,min_phi,max_phi,min_rho,max_rho,spacing_theta,spacing_phi,spacing_rho,photon_count,camera,spectrum,scatter = False,origin = [0,0,0],origin_metal = [0,0,0]):
    # generate angle pairs on a sphere
    thetas, phis, rhos = projection_matrix.generate_uniform_angels_modified(min_theta, max_theta, min_phi, max_phi, min_rho, max_rho, spacing_theta, spacing_phi,spacing_rho)
    # generate projection matrices from angles
    proj_mats = projection_matrix.generate_projection_matrices_from_values(camera.source_to_detector_distance, camera.pixel_size, camera.pixel_size, camera.sensor_width, camera.sensor_height, camera.isocenter_distance, phis, thetas, rhos)

    ####
    #Recommend origin settings
    # origin = [70,0,10]
    

    #load CT volume
    trun=1000
    # ori_CT_volume, CT_volume, CT_materials, CT_voxel_size = load_nifti_CT(CT_volume_path, segmented_files_path, truncate=[[0,None],[0,None],[0,trun]], use_thresholding_segmentation=True)

    # ori_CT_volume, CT_volume, CT_materials, CT_voxel_size = load_dicom_CT(CT_volume_path, fixed_slice_thinckness=1.0, truncate=[[0,None],[0,None],[0,trun]], use_thresholding_segmentation=True)
    ori_CT_volume, CT_volume, CT_materials, CT_voxel_size = load_dicom_CT_new(CT_volume_path, segmented_files_path, fixed_slice_thinckness=1.0,
                                                                          truncate=[[0, None], [0, None], [0, trun]],
                                                                          use_thresholding_segmentation=True)
    # origin = np.round(get_origin.get_origin(segmented_files_path)*CT_voxel_size).astype(int)

    ##forward CT project densities of CT materials
    forward_projections = projector.generate_projections(proj_mats, CT_volume, CT_materials, origin, CT_voxel_size, camera.sensor_width, camera.sensor_height, mode="linear", max_blockind=200, threads=8)

    forward_projections, metal_voxel_size= add_metal_into_foward_projections(forward_projections, proj_mats,ori_CT_volume,CT_volume,CT_voxel_size,CT_materials,origin,camera,metal1_path,origin_metal)

    # forward_projections, metal_voxel_size= add_metal_into_foward_projections(forward_projections, proj_mats, ori_CT_volume, CT_volume, CT_voxel_size,CT_materials, origin, camera, metal2_path, origin_metal)

    images, photon_prob = mass_attenuation.calculate_intensity_from_spectrum(forward_projections, spectrum)

    #add scatter
    # if scatter:
    #     scatter_net = add_scatter.ScatterNet()
    #     scatter = scatter_net.add_scatter(images, camera)
    #     photon_prob *= 1 + scatter/images
    #     images += scatter

    #transform to collected energy in keV per cm^2
    # images = images * (photon_count / (camera.pixel_size * camera.pixel_size))

    #add poisson noise
    # images = add_noise(images, photon_prob, photon_count)

    #use negative film
    images = 256*np.ones(images.shape)-images

    #auto-label generation (should be fine-tuned in advance)
    radiograph, label_set = draw_box.annotate(camera=camera,image=images[0,:,:],origin_metal=origin_metal,metal_voxel_size=metal_voxel_size,draw_box=False)
    plt.imshow(radiograph, cmap="gray")
    plt.axis('off')  # Remove axis
    plt.tight_layout(pad=0)  # Remove padding around the image
    plt.show()

def main():
    #2x2 binning
    
    #(raw) camera = Camera(sensor_width = 512, sensor_height = 512, pixel_size = 0.62, source_to_detector_distance = 1200, isocenter_distance = 450)
    # camera = Camera(sensor_width=1280, sensor_height=960, pixel_size=0.31, source_to_detector_distance=1100,isocenter_distance=800)
    camera = Camera(sensor_width=512, sensor_height=512, pixel_size=0.1, source_to_detector_distance=400,
                    isocenter_distance=1800) #calibrated
    #source_to_detector_distance the lower, the vision is more broad,
    #4x4 binning
    #(raw) camera = Camera(sensor_width=620, sensor_height=480, pixel_size=0.62, source_to_detector_distance=1200,isocenter_distance=800)

    ####
    #define the path to your dicoms here or use the simple phantom from the code above
    ####
    #CT_volume_path = r".\your_dicom_directory\\"
    # CT_volume_path = r"./Radiogenomics/CT"
    CT_volume_path = r"./Radiogenomics/3000566"
    segmented_files_path = r"./Radiogenomics/3000566/segmented_files"
    metal1_path =  r"./sample_metal_volume/rfo_dcm_models/COVIDIEN_electrode.dcm"
    metal2_path = r"./sample_metal_volume/rfo_dcm_models/COVIDIEN_electrode.dcm"

    save_path = r"./test/totalseg"
    min_theta = 180 #angle between the camera vector's shadow on xoy plane and y+, counterclockwise on xoy plane going larger (vision from right(+)/left(-) side)
    max_theta = 181
    min_phi = -175 #camera self-rotate around y axis, clockwise on xoz plane going larger
    max_phi = 181
    min_rho = 90  #angle between the camera vector's shadow on yoz plane and y+, clockwise on yoz plane going larger  (vision up(+)/down(-))
    max_rho = 181
    spacing_theta = 360
    spacing_phi = 360
    spacing_rho = 360
    photon_count = 100000
    # origin [0,0,0] corresponds to the center of the volume
    origin = [5,0,30] #used for 3000566   #right, backward, down
    origin_metal = [60,40,90]
    # origin = [10,100,-250] #for /CT
    # origin,a = get_origin.get_origin(segmented_files_path)
    spectrum = spectrum_generator.SPECTRUM120KV_AL43


    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    generate_projections_on_sphere(CT_volume_path,metal1_path,metal2_path,segmented_files_path,save_path,min_theta,max_theta,min_phi,max_phi,min_rho,max_rho,spacing_theta,spacing_phi,spacing_rho,photon_count,camera,spectrum,origin=origin,scatter=False,origin_metal=origin_metal)


if __name__ == "__main__":
    main()