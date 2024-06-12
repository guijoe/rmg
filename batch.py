import os
import sys
import time
import numpy as np
import pandas as pd
import src.rmg as rmg
import src.sphm as sh
import multiprocessing
import src.meshutils as mutils




def process_image(input_folder, image_file, subdivisions, iterations, neta, cell_surfaces, previous_meshes, t):
    #print(image_file)
    image_path = input_folder + image_file
    cell_meshes = rmg.generate_regular_meshes(image_path, subdivisions, iterations, neta, cell_surfaces, previous_meshes)
    previous_meshes = cell_meshes

    # Write cell meshes to an OBJ file
    obj_image_file = output_folder + os.path.splitext(image_file)[0] + '.obj'
    mutils.save_meshes_obj(cell_meshes, t, obj_image_file)
    t=t+1
    return t

subdivisions = int(sys.argv[2]) # Number of subdivision of the icosahedron. N=2 will yield meshes with 162 points, N=k ... with 10*4^k + 2 points
iterations = int(sys.argv[3]) # Number of iterations of each update loop of the level set shape matching scheme
cell_surfaces = False if sys.argv[4] == 'False' or sys.argv[4] == '0' else True # If true, computes cell surfaces, if false, computes embryonic surface. True by default

# Specify the folder containing input and output folders
input_folder = sys.argv[1]
output_folder = input_folder.split("nii", 1)[0] + "obj" + input_folder.split("nii", 1)[1]
output_folder = output_folder + "cells/" if cell_surfaces else output_folder + "emb/"
if not os.path.exists(output_folder): os.makedirs(output_folder)




# List all files in the folder
file_list = os.listdir(input_folder)
nii_images_list = [f for f in sorted(file_list) if f.endswith('.nii') or f.endswith('.inr')]

neta = [5 if i % 5 == 0 else 5 for i in range(iterations)] # 'Learning rate' for each update loop. Alternate learning tends to converge faster and is more robust
    
start_time_folder = time.time()

folder_path = root_folder + folder + "/"

cell_surfaces = True # If True, computes cellular surfaces, if False, computes embryonic surface
subdivisions = 2
iterations = 125

if not cell_surfaces:
    subdivisions = 3

#command += "python3 main.py " + (folder_path) + " " + str(subdivisions) + " " + str(iterations) + " " + str(cell_surfaces) + " &"
os.system("python3 main.py " + (folder_path) + " " + str(subdivisions) + " " + str(iterations) + " " + str(cell_surfaces))

end_time_folder = time.time()

elapsed_time = end_time_folder - start_time_folder
print(f"Execution time for {folder}: {elapsed_time:.6f} seconds")

end_time_batch = time.time()

print(f"Total execution time: {end_time_batch - start_time_batch:.6f} seconds")