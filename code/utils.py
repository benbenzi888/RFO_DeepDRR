import numpy as np
import PIL.Image as Image
from datetime import datetime
import os
import pickle
import csv

def image_saver(image, prefix, path, projection_count, rfo_name=""):

    image_pil = image
    # image_pil.save(path + "\\" +prefix + str(i).zfill(5) + ".tiff")
    if image_pil.dtype != np.uint8:

        image_pil = 255 * (image_pil - image_pil.min()) / (image_pil.max() - image_pil.min())
        image_pil = Image.fromarray(image_pil.astype(np.uint8))

    # image_pil.save(path + "/" + prefix + "_"+ str(projection_count).zfill(6) + rfo_name + "_" + ".tiff")
    image_pil.save(path + "/" + str(projection_count).zfill(5) + ".tiff")
    return True

def annotation_creator(data_index, add_rfo_flag, label=None):
    label_file_path = "./produced_dataset/"

    file_name = os.path.join(label_file_path+ "full500.csv")
    if add_rfo_flag:
        new_row = [str(data_index).zfill(5) + '.tiff', " ".join(map(str, label))]  # Customize the new row as needed
    else:
        new_row = [str(data_index).zfill(5) + '.tiff']
    # Check if the file exists
    if os.path.exists(file_name):
        print(f"'{file_name}' exists. Adding a new row.")
        # Open the file in append mode and add the new row
        with open(file_name, mode='a', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(new_row)
    else:
        print(f"'{file_name}' does not exist. Creating the file.")
        # Create the file and write the new row
        with open(file_name, mode='w', newline='') as file:
            csv_writer = csv.writer(file)
            # Optionally, write a header row first
            header = ["image_name","annotation"]  # Customize headers
            csv_writer.writerow(header)
            csv_writer.writerow(new_row)

    print(f"Row added to '{file_name}' successfully.")

def param_saver(thetas, phis, proj_mats, camera, origin, photons, spectrum, prefix, save_path):
    i0 = np.sum(spectrum[:,0]*(spectrum[:,1]/np.sum(spectrum[:,1])))/1000
    data = {"date": datetime.now(), "thetas": thetas, "phis": phis, "proj_mats": proj_mats, "camera": camera, "origin": origin, "photons": photons, "spectrum": spectrum, "I0":i0}
    with open(os.path.join(save_path,prefix+'.pickle'), 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    return True

class Camera():
    def __init__(self,sensor_width,sensor_height,pixel_size,source_to_detector_distance,isocenter_distance):
        self.sensor_width = sensor_width
        self.sensor_height = sensor_height
        self.pixel_size = pixel_size
        self.source_to_detector_distance = source_to_detector_distance
        self.isocenter_distance = isocenter_distance