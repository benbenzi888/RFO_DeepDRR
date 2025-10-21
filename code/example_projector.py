
import projector
import projection_matrix
from load_dicom import load_dicom
from analytic_generators import add_noise
import mass_attenuation_gpu as mass_attenuation
import spectrum_generator
import add_scatter
from utils import image_saver, Camera, param_saver
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import rotate
from draw_box import bresenham_line

def generate_projections_on_sphere(volume_path,save_path,min_theta,max_theta,min_phi,max_phi, min_rho, max_rho, spacing_theta,spacing_phi, spacing_rho, photon_count,camera,spectrum,scatter = False,origin = [0,0,0]):
    # generate angle pairs on a sphere
    # thetas, phis = projection_matrix.generate_uniform_angels(min_theta, max_theta, min_phi, max_phi, spacing_theta, spacing_phi)
    thetas, phis, rho = projection_matrix.generate_uniform_angels_modified(min_theta, max_theta, min_phi, max_phi, min_rho, max_rho, spacing_theta, spacing_phi, spacing_rho)
    # generate projection matrices from angles
    proj_mats = projection_matrix.generate_projection_matrices_from_values(camera.source_to_detector_distance, camera.pixel_size, camera.pixel_size, camera.sensor_width, camera.sensor_height, camera.isocenter_distance, phis, thetas, rho)

    ####
    #Use this if you have a volume
    ####
    #load and segment volume
    # volume, materials, voxel_size = load_dicom(volume_path, use_thresholding_segmentation=True)

    ####
    #Otherwise use this simple phantom for test
    ####
    #start of phantom
    volume = np.zeros((100,100,100),dtype = np.float32)
    volume[30:70, 30:70, 30:70] = 2

    # volume[60:80, 60:80, 60:80] = 1
    materials = {}
    materials["air"] = volume == 0
    materials["soft tissue"] = volume == 1
    materials["bone"] = volume == 2
    voxel_size = np.array([2,2,2],dtype=np.float32)
    #end of phantom

    #rotation
    # new_volume = np.zeros((1000,1000,1000),dtype=np.float32)
    # new_volume[244:244+512,244:244+512,450:450+133]=volume
    # volume = rotate(new_volume, angle=20, axes=(1,0), reshape=False)


    #forward project densities
    forward_projections = projector.generate_projections(proj_mats, volume, materials, origin, voxel_size, camera.sensor_width, camera.sensor_height, mode="linear", max_blockind=150, threads=8)
    # calculate intensity at detector (images: mean energy one photon emitted from the source deposits at the detector element, photon_prob: probability of a photon emitted from the source to arrive at the detector)
    images, photon_prob = mass_attenuation.calculate_intensity_from_spectrum(forward_projections, spectrum)
    #add scatter
    if scatter:
        scatter_net = add_scatter.ScatterNet()
        scatter = scatter_net.add_scatter(images, camera)
        photon_prob *= 1 + scatter/images
        images += scatter

    #transform to collected energy in keV per cm^2
    # images = images * (photon_count / (camera.pixel_size * camera.pixel_size))

    #add poisson noise
    images = add_noise(images, photon_prob, photon_count)

    #save images
    image_saver(images, "DRR", save_path)
    #save parameters
    param_saver(thetas, phis, proj_mats, camera, origin, photon_count,spectrum, "simulation_data", save_path)

    #show result
    a = images[0,:,:]
    # vertex_set=np.array([[30,35,40,1],[30,65,40,1],[70,35,40,1],[70,65,40,1]])  # + origin
    # Transform_matrix = np.array([[0, 1.2, 0, 67.3],[1.2, 0, 0, 67.3],[0, 0, 0, 1]])
    # mapping_set = np.round(Transform_matrix @ np.transpose(vertex_set)).T.astype(int)
    # mapping_set = np.delete(mapping_set,2,1)
    # central_point = np.array([127,127])
    # mapping_set = (np.tile(central_point,(mapping_set.shape[0],1)) + (mapping_set-np.tile(central_point,(mapping_set.shape[0],1)))*camera.source_to_detector_distance/300*800/camera.isocenter_distance*0.31/camera.pixel_size*voxel_size[0]/1).astype(int)
    # result = tuple(map(tuple, mapping_set))
    # for start in result:
    #     for end in result:
    #         line_points = bresenham_line(start, end)
    #         for x, y in line_points:
    #             a[x, y] = 200  # Mark the line
    plt.imshow(a, cmap="gray")
    plt.show()


def main():
    #2x2 binning
    # camera = Camera(sensor_width = 1240, sensor_height = 960, pixel_size = 0.31, source_to_detector_distance = 1200, isocenter_distance = 800)
    camera = Camera(sensor_width=256, sensor_height=256, pixel_size=0.31, source_to_detector_distance=300, isocenter_distance=800)
    # sensor_width and sensor_height measures the size of the camera window, which is going larger with vision going larger, the same object
    # going smaller
    # pixel_size measures the size of detector, which is larger with vision going larger, the same object going smaller
    # source_to_detector_distance measures the distance between camera and source, which is going larger with projection getting larger, the same
    # camera receiving more detailed information
    # isocenter_distance measures the distance between source and object, which is going larger with projection getting smaller. the same
    #camera receiving more general information.
    #4x4 binning
    # camera = Camera(sensor_width=620, sensor_height=480, pixel_size=0.62, source_to_detector_distance=1200,isocenter_distance=800)
    ####
    #define the path to your dicoms here or use the simple phantom from the code above
    ####
    # dicompath = r"./compare/modified/"
    dicompath = r"./Radiogenomics/CT/"
    # / home / benben / Projects / deepdrr - 0.1 / potential_dataset / StageII - Colorectal - CT - 001
    save_path = r"./Results_picture/raw/"
    #suppose a xyz coordinates, camera going right is x+, going deeper is y+, going higher is z+
    min_theta = 0 #angle between the camera vector's shadow on xoy plane and y+, counterclockwise on xoy plane going larger (vision from right(+)/left(-) side)
    max_theta = 180
    min_phi = 0 #camera self-rotate around y axis, clockwise on xoz plane going larger
    max_phi = 181
    min_rho = 0
    #angle between the camera vector's shadow on yoz plane and y+, clockwise on yoz plane going larger  (vision up(+)/down(-))
    max_rho = 181
    spacing_theta = 360
    spacing_phi =360
    spacing_rho = 360
    photon_count = 100000
    # origin [0,0,0] corresponds to the center of the volume
    # origin = [0,200,50]  # [camera go x-, camera go z+, camera go y-]
    origin = [20, 20, 0]
    # origin = [0, 30, 0]
    spectrum = spectrum_generator.SPECTRUM120KV_AL43
    #lateral
    # min_theta = 90 #angle between the camera vector's shadow on xoy plane and y+, counterclockwise on xoy plane going larger (vision from right(+)/left(-) side)
    # max_theta = 180
    # min_phi = 0 #camera self-rotate around y axis, clockwise on xoz plane going larger
    # max_phi = 181
    # min_rho = 0 #angle between the camera vector's shadow on yoz plane and y+, clockwise on yoz plane going larger  (vision up(+)/down(-))
    # max_rho = 180
    # spacing_theta = 60
    # spacing_phi = 60
    # spacing_rho = 60
    # photon_count = 100000
    # #origin [0,0,0] corresponds to the center of the volume
    # origin = [-300,-20,0]  # [camera go x-, camera go z+, camera go y-]
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    generate_projections_on_sphere(dicompath,save_path,min_theta,max_theta,min_phi,max_phi,min_rho,max_rho,spacing_theta,spacing_phi,spacing_rho,photon_count,camera,spectrum,origin=origin,scatter=False)

if __name__ == "__main__":
    main()
