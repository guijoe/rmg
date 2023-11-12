import os
import sys
import numpy as np
import pandas as pd
import src.rmg as rmg
import src.sphm as sh
import multiprocessing
import src.meshutils as mutils

subdivisions = int(sys.argv[2]) # Number of subdivision of the icosahedron. N=2 will yield meshes with 162 points, N=k ... with 10*4^k + 2 points
iterations = int(sys.argv[3]) # Number of iterations of each update loop of the level set shape matching scheme
cell_surfaces = False if sys.argv[4] == 'False' or sys.argv[4] == '0' else True # If true, computes cell surfaces, if false, computes embryonic surface. True by default

# Specify the folder containing input and output folders
input_folder = sys.argv[1]
output_folder = input_folder.split("nii", 1)[0] + "obj" + input_folder.split("nii", 1)[1]
output_folder = output_folder + "cells/" if cell_surfaces else output_folder + "emb/"
if not os.path.exists(output_folder): os.makedirs(output_folder)
if not os.path.exists(output_folder + "obj"): os.makedirs(output_folder + "obj")
#print(input_folder, output_folder, cell_surfaces)

# List all files in the folder
file_list = os.listdir(input_folder)
nii_images_list = [f for f in sorted(file_list) if f.endswith('.nii') or f.endswith('.inr')]

neta = [1 if i % 5 == 0 else 1 for i in range(iterations)] # 'Learning rate' for each update loop. Alternate learning tends to converge faster and is more robust

""" def process_image(input_folder, image_file, subdivisions, iterations, neta, cell_surfaces, previous_meshes, t):
    #print(image_file)
    image_path = input_folder + image_file
    cell_meshes = rmg.generate_regular_meshes(image_path, subdivisions, iterations, neta, cell_surfaces, previous_meshes)
    previous_meshes = cell_meshes

    # Write cell meshes to an OBJ file
    obj_image_file = output_folder + os.path.splitext(image_file)[0] + '.obj'
    mutils.save_meshes_obj(cell_meshes, t, obj_image_file)
    t=t+1
    return t

t = 1
arguments = []
previous_meshes = []

batch_length = multiprocessing.cpu_count()//2

for i in range(0, len(nii_images_list), batch_length):
    if i+batch_length < len(nii_images_list):
        batch_nii_images_list = nii_images_list[i:i+batch_length]
    else:
        batch_nii_images_list = nii_images_list[i:len(nii_images_list)]

    #nii_images_list = nii_images_list[0:10]
    print("Number of images: " , len(batch_nii_images_list))
    for image_file in batch_nii_images_list:
        image_path = input_folder + image_file
        arguments += [(input_folder, image_file, subdivisions, iterations, neta, cell_surfaces, previous_meshes, t)]
        t=t+1
    #print(arguments[0])

    if __name__ == '__main__':
        num_processes = multiprocessing.cpu_count()  # Adjust the number of processes according to your system resources
        pool = multiprocessing.Pool(processes=num_processes)

        results = pool.starmap(process_image, arguments)

        pool.close()
        pool.join() """







t = 1
Lmax = 20
previous_meshes = []
Sl = np.zeros((0, Lmax+1))
Sl = []
frames = []
cell_ids = []

nii_images_list = nii_images_list[0:1]
for image_file in nii_images_list:
    image_path = input_folder + image_file

    # Generate the meshes from images
    cell_meshes = rmg.generate_regular_meshes(image_path, subdivisions, iterations, neta, cell_surfaces, previous_meshes)
    previous_meshes = cell_meshes

    # Write cell meshes to an OBJ file
    obj_image_file = output_folder + "obj/" + os.path.splitext(image_file)[0] + '.obj'
    rmg.save_meshes_obj(cell_meshes, t, obj_image_file)

    # Compute Spherical harmonics representation of the (normalized) radius field on all meshes
    """ for i, mesh in enumerate(cell_meshes):
        r, r_norm, theta, phi, dA, _,_,_ = mutils.compute_vertex_properties(mesh)
        f = r_norm
        fi = np.zeros(r.shape)

        frames += [t-1]
        cell_ids += [i]
        Sl += [sh.compute_sphm_inv_rep(f,fi,theta,phi,dA,Lmax)] """   
    t=t+1

""" # Save spherical harmonics representation to csv
columns = [f"S{i}" for i in range(Lmax+1)]
df = pd.DataFrame(Sl)
df.columns = columns
df["fr"] = frames
df["id"] = cell_ids
print(df.head())
df.to_csv(output_folder + '/cells_rot_inv_rep.csv', index=False) """