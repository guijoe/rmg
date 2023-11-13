import time
import morphonet
import numpy as np
import vtkmodules.all as vtk
from scipy.spatial import KDTree
from skimage import morphology,measure
import src.meshutils as mutils


# Computes the minimum distance to the pixel cloud and the indices of the closest pixels
def min_distance_index(vertices, surface_pixels):
    min_distance = np.zeros(len(vertices))
    min_index = np.zeros(len(vertices))
    for i in range(len(vertices)):
        distances_from_i = np.sqrt(np.sum((surface_pixels - vertices[i])**2, axis=1))
        min_distance[i] = np.min(distances_from_i)
        min_index[i] = np.argmin(distances_from_i)
    return min_distance,min_index

def min_distance_index2(vertices, surface_pixels_per_vertex, nearest_indices):
    min_distance = np.zeros(len(vertices))
    min_index = np.zeros(len(vertices))
    for i in range(len(vertices)):
        distances_from_i = np.sqrt(np.sum((surface_pixels_per_vertex[i] - vertices[i])**2, axis=1))
        min_distance[i] = np.min(distances_from_i)
        min_index[i] = nearest_indices[i][int(np.argmin(distances_from_i))]
    return min_distance,min_index

# Saves mesh in obj format

def generate_regular_meshes(image_file):

    image = morphonet.tools.imread(image_file)
    labeled_image = measure.label(image)
    regions = measure.regionprops(labeled_image)

    cell_surface_masks = []

    embryo_mask = labeled_image == 1
    embryo_surface_mask = cell_mask ^ morphology.erosion(embryo_mask, morphology.cube(3))
    
    cell_meshes = []
    nb_regions = len(regions)+1
    cell_regions_range = range(2, nb_regions)
    # Iterate through the labeled regions (excluding background)
    for cl in cell_regions_range:#nb_regions):
        start_time_cell = time.time()
    
        #### Comupte surface pixels
        cell_mask = labeled_image == cl
        cell_surface_mask = cell_mask ^ morphology.erosion(cell_mask, morphology.cube(3))

        cell_surface_masks += [cell_surface_mask]
        
    overlap_image = np.sum(cell_surface_masks, axis=0)

    # Find points where three or more regions overlap
    overlap_points = np.where(overlap_image >= 3)

    # Visualize the results
    #plt.imshow(gray_image, cmap=plt.cm.gray)

    # Highlight overlap points
    #plt.plot(overlap_points[1], overlap_points[0], 'r.', markersize=5, label='Overlap Points')

    #plt.legend()
    #plt.title('Image with Overlap Points')
    #plt.show()

    return overlap_points
