import os
import time
import morphonet
import numpy as np
import pandas as pd
import multiprocessing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist
from skimage import morphology,io,measure,color,filters, img_as_float



#def get_cell_surface_mask(cell_mask):
#    cell_surface_mask = cell_mask ^ morphology.dilation(morphology.dilation(cell_mask))
#    return cell_surface_mask

start_time = time.time()

frames = []
corners = []
input_folder = "nii/Astec-Pm1/"
file_list = os.listdir(input_folder)
inr_images_list = [f for f in sorted(file_list) if f.endswith('.inr')]
inr_images_list = inr_images_list[0:3]
print(inr_images_list)

for t,im in enumerate(inr_images_list):
    start_time_frame = time.time()
    image = morphonet.tools.imread(input_folder+im)

    #image = morphonet.tools.imread(image)
    labeled_image = measure.label(image)
    regions = measure.regionprops(labeled_image)

    cell_surface_masks = []

    embryo_mask = labeled_image == 1
    embryo_surface_mask = embryo_mask ^ morphology.erosion(embryo_mask, morphology.cube(3))

    cell_meshes = []
    nb_regions = len(regions)+1
    cell_regions_range = range(2, nb_regions)
    #network_mask = np.ones(image.shape, dtype=np.uint8)
    # Iterate through the labeled regions (excluding background)
    for cl in cell_regions_range:#nb_regions):
        start_time_cell = time.time()

        #### Comupte surface pixels
        cell_mask = labeled_image == cl
        cell_surface_mask = cell_mask ^ morphology.dilation(cell_mask, morphology.cube(3)) 
        #morphology.dilation

        #network_mask = network_mask & cell_surface_mask
        cell_surface_masks += [cell_surface_mask]

        end_time_cell = time.time()
        print(f"Total processing time for cell {cl}/{len(cell_regions_range)} at frame {t}: {end_time_cell - start_time_cell}")
                
    overlap_image = np.sum(cell_surface_masks, axis=0)
    overlap_image += embryo_surface_mask

    # Find points where three or more regions overlap
    overlap_points = np.where(overlap_image >= 4)
    print("Overlaping points found: ", len(overlap_points[0]))

    points = np.array([np.array([overlap_points[0][i],overlap_points[1][i],overlap_points[2][i]]) for i in range(len(overlap_points[0]))])

    threshold = 5
    distances = cdist(points, points)
    filtered_distances = np.where(distances < threshold)
    pairs = np.array([np.array([filtered_distances[0][i], filtered_distances[1][i]]) for i in range(len(filtered_distances[0]))])
    for i in range(len(np.unique(filtered_distances[0]))):
        j_s = pairs[pairs[:, 0] == i, 1]
        for j in j_s:
            if j>i:
                pairs[pairs[:, 0] == j, 0] = i       
    
    cnrs = np.array([points[i] for i in np.unique(pairs[:,0])])
    print(cnrs.shape)
    
    print("Sum done")

    frames += [t for _ in range(cnrs.shape[0])] #frames.append(np.ones(cnrs.shape[0]) * t)
    corners += [cnrs[i] for i in range(len(cnrs))]
    print(len(corners), len(frames))
    end_time_frame = time.time()
    print(f"Total processing time for frame {t}/{len(inr_images_list)}: {end_time_frame - start_time_frame}")

columns = ["x", "y", "z"]
df = pd.DataFrame(corners)
df.columns = columns
df["fr"] = frames
#df.insert(0, "fr", frames)
df.to_csv(input_folder + "/corners.csv", sep=' ')

end_time = time.time()
print(f"Total processing time for frame {t}/{len(inr_images_list)}: {end_time - start_time}")
