import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom import Sequence
import vtk
from vtk.util import numpy_support

# Step 1: Read the .obj file and convert it into voxel data
def load_obj_as_solid_pixels(obj_file, resolution=(100, 100, 100)):
    # Read the OBJ file
    reader = vtk.vtkOBJReader()
    reader.SetFileName(obj_file)
    reader.Update()

    # Get the PolyData from the OBJ file
    polydata = reader.GetOutput()

    # Define the bounding box
    bounds = polydata.GetBounds()

    # Set the spacing for the voxelizer
    spacing = [(bounds[1] - bounds[0]) / resolution[0],
               (bounds[3] - bounds[2]) / resolution[1],
               (bounds[5] - bounds[4]) / resolution[2]]

    # Create the grid for volume data (ImageData)
    white_image = vtk.vtkImageData()
    white_image.SetSpacing(spacing)

    # Set the extent of the volume data
    white_image.SetDimensions(resolution)
    white_image.SetExtent(0, resolution[0] - 1, 0, resolution[1] - 1, 0, resolution[2] - 1)
    white_image.SetOrigin(bounds[0], bounds[2], bounds[4])

    # Initialize all voxels with a value (e.g., 255)
    white_image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    for i in range(resolution[0]):
        for j in range(resolution[1]):
            for k in range(resolution[2]):
                white_image.SetScalarComponentFromFloat(i, j, k, 0, 1)

    # Create a PolyDataToImageStencil filter to convert PolyData to voxels
    pol2stenc = vtk.vtkPolyDataToImageStencil()
    pol2stenc.SetInputData(polydata)
    pol2stenc.SetOutputSpacing(spacing)
    pol2stenc.SetOutputOrigin(bounds[0], bounds[2], bounds[4])
    pol2stenc.SetOutputWholeExtent(white_image.GetExtent())
    pol2stenc.Update()

    # Create an ImageStencil filter to set the voxel values inside PolyData to 0
    imgstenc = vtk.vtkImageStencil()
    imgstenc.SetInputData(white_image)
    imgstenc.SetStencilConnection(pol2stenc.GetOutputPort())
    imgstenc.ReverseStencilOff()
    imgstenc.SetBackgroundValue(0)  # Set the outside to 0
    imgstenc.Update()

    # Extract the volume data and convert it to a NumPy array
    vtk_data = imgstenc.GetOutput()
    dims = vtk_data.GetDimensions()
    vtk_array = vtk.util.numpy_support.vtk_to_numpy(vtk_data.GetPointData().GetScalars())

    # Reshape the NumPy array to 3D
    solid_pixels = vtk_array.reshape(dims[2], dims[1], dims[0])
    solid_pixels = np.transpose(solid_pixels, (2, 1, 0))  # Reorder axes to (X, Y, Z)

    return solid_pixels


# Step 2: Create a DICOM file and write header information, including (5200, 9229) and (5200, 9230)
# Modified DICOM file creation part to handle 3D pixel data

def create_dicom_file(output_file, pixel_array, pixel_spacing, image_orientation):
    # Create file meta information
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    file_meta.ImplementationClassUID = "1.2.826.0.1.3680043.2.1143.107.104.103.115.2.8.5.111.124.113"

    # Create main dataset
    ds = Dataset()
    ds.file_meta = file_meta
    ds.is_little_endian = True
    ds.is_implicit_VR = True

    # Basic header information
    ds.SOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.StudyDate = "20190522"
    ds.StudyTime = "155321.542139"
    ds.Modality = "OT"
    ds.PatientName = "Patient^Name"
    ds.PatientID = "123456"

    # Set Rows and Columns, and NumberOfFrames
    depth, rows, columns = pixel_array.shape  # For 3D data, shape returns three dimensions
    ds.Rows = rows
    ds.Columns = columns
    ds.NumberOfFrames = str(depth)  # Use depth as NumberOfFrames

    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16  # Confirm the bit depth
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0  # Unsigned integer
    pixel_array = pixel_array.astype(np.uint16)
    # Flatten the 3D pixel data into 2D
    ds.PixelData = pixel_array.tobytes()

    # Step 3: Add (5200, 9229) Shared Functional Groups Sequence
    shared_functional_groups = Dataset()

    # Add Image Orientation (0020, 0037)
    plane_orientation = Dataset()
    plane_orientation.ImageOrientationPatient = image_orientation
    plane_orientation_sequence = Sequence([plane_orientation])
    shared_functional_groups.PlaneOrientationSequence = plane_orientation_sequence

    # Add Pixel Spacing (0028, 0030)
    pixel_measures = Dataset()
    pixel_measures.PixelSpacing = pixel_spacing
    pixel_measures_sequence = Sequence([pixel_measures])
    shared_functional_groups.PixelMeasuresSequence = pixel_measures_sequence

    ds.SharedFunctionalGroupsSequence = Sequence([shared_functional_groups])

    # Step 4: Add (5200, 9230) Per-Frame Functional Groups Sequence and SliceLocation
    per_frame_sequence = []
    for i in range(depth):
        frame_functional_groups = Dataset()

        # Per-frame position information (0020, 0032)
        plane_position = Dataset()
        plane_position.ImagePositionPatient = [0, 0, i * pixel_spacing[1]]  # Assuming the slice spacing is related to the second pixel_spacing value
        plane_position_sequence = Sequence([plane_position])
        frame_functional_groups.PlanePositionSequence = plane_position_sequence

        # Add SliceLocation (0020, 1041)
        ds.SliceLocation = i * pixel_spacing[1]  # Set Z-axis position as SliceLocation

        # Add Instance Number (0020, 0013) -- Frame number functionality
        ds.InstanceNumber = i + 1  # Frame numbering starts from 1

        per_frame_sequence.append(frame_functional_groups)

    ds.PerFrameFunctionalGroupsSequence = Sequence(per_frame_sequence)

    # Write the DICOM file
    pydicom.dcmwrite(output_file, ds)


# Test code

# Assume OBJ file and DICOM output path
# obj_file = './rfo_models/DEROYAL_cotton_ball/#2_tmpm3zyq9pd.obj'
obj_file = '../rfo_obj_models/ivc_filter_2.obj'
output_dcm_file = '#2_ivc_filter.dcm'

# Load the OBJ file as pixel data
pixel_data = load_obj_as_solid_pixels(obj_file)

if pixel_data is not None:
    # Set the DICOM frame count, pixel spacing, and image orientation
    pixel_spacing = [0.5, 0.5]  # Set pixel spacing, typically in millimeters
    # pixel_spacing = [2, 2]  # Set pixel spacing, typically in millimeters
    # image_orientation = [1, 0, 0, 0, 1, 0]  # Set image orientation, here it’s aligned to patient axial plane
    # image_orientation = [1, 0, 0, 0, 0, 1]  # Set image orientation, here it’s aligned to patient coronal plane
    image_orientation = [0, 1, 0, 0, 0, 1]  # Set image orientation, here it’s aligned to patient axial plane

    # Generate the DICOM file
    create_dicom_file(output_dcm_file, pixel_data, pixel_spacing, image_orientation)
    print(f"DICOM file saved to {output_dcm_file}")
else:
    print("Failed to load OBJ file.")

