import numpy as np

def bresenham_line(point1, point2):
    x1, y1 = point1
    x2, y2 = point2

    points = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while True:
        # points.append((x1, y1))
        if x1 == x2 and y1 == y2:
            break
        if 0 < x1 < 255 and 0 < y1 < 255:
            points.append((x1, y1))
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy

    return points

def annotate(camera,image,origin_metal=[0,0,0],metal_voxel_size=[1,1,1],draw_box = False):
    i, j = 1, 2
    origin_metal[i], origin_metal[j] = origin_metal[j], origin_metal[i]
    vertex_set=np.array([[60,60,60,1],[60,-60,-60,1],[-60,60,60,1],[-60,-60,-60,1]])+np.tile(np.append(origin_metal,0),(4,1))/metal_voxel_size[0] # + origin
    # Transform_matrix = np.array([[-0.285, 0, 0, 127],[0, 0.285, 0, 127],[0, 0, 0, 1]])
    Transform_matrix = np.array([[-0.282, 0, 0, 127], [0, 0.282, 0, 127], [0, 0, 0, 1]])
    mapping_set = np.round(Transform_matrix @ np.transpose(vertex_set)).T.astype(int)
    mapping_set = np.delete(mapping_set,2,1)#used for 3000566  #re
    central_point = np.array([127,127])
    mapping_set = (np.tile(central_point,(mapping_set.shape[0],1)) + (mapping_set-np.tile(central_point,(mapping_set.shape[0],1)))*camera.source_to_detector_distance/170*1200/camera.isocenter_distance*0.1/camera.pixel_size*metal_voxel_size[0]/0.2).astype(int)
    # print(mapping_set)
    if draw_box:
        result = tuple(map(tuple, mapping_set))
        for start in result:
            for end in result:
                line_points = bresenham_line(start, end)
                for x, y in line_points:
                    image[x, y] = 256  # Mark the line
    print(f"annotation:0 {mapping_set[3,1]} {mapping_set[0,0]} {mapping_set[0,1]} {mapping_set[3,0]}" )
    label_set = [0, mapping_set[3,1], mapping_set[0,0], mapping_set[0,1], mapping_set[3,0]]
    return image, label_set