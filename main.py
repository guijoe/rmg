import os
import sys
import numpy as np
import pandas as pd
import src.rmg as rmg
import src.sphm as sh
import multiprocessing
from itertools import repeat
import src.meshutils as mutils

subdivisions = int(sys.argv[2]) # Number of subdivision of the icosahedron. N=2 will yield meshes with 162 points, N=k ... with 10*4^k + 2 points
iterations = int(sys.argv[3]) # Number of iterations of each update loop of the level set shape matching scheme
cell_surfaces = False if sys.argv[4] == 'False' or sys.argv[4] == '0' else True # If true, computes cell surfaces, if false, computes embryonic surface. True by default

# Specify the folder containing input and output folders
input_folder = sys.argv[1]
output_folder = input_folder.split("nii", 1)[0] + "obj" + input_folder.split("nii", 1)[1]
output_folder = output_folder + "cells/" if cell_surfaces else output_folder + "emb/"
if not os.path.exists(output_folder): os.makedirs(output_folder)
#if not os.path.exists(output_folder + "obj"): os.makedirs(output_folder + "obj")
#print(input_folder, output_folder, cell_surfaces)

# List all files in the folder
file_list = os.listdir(input_folder)
nii_images_list = [f for f in sorted(file_list) if f.endswith('.nii') or f.endswith('.inr')]
nii_images_list = nii_images_list[0:3]

#print(nii_images_list)


neta = [10 if i % 5 == 0 else 10 for i in range(iterations)] # 'Learning rate' for each update loop. Alternate learning tends to converge faster and is more robust

t = 1
Lmax = 20
#Sl = np.zeros((0, Lmax+1))
Sl = []
frames = []
cell_ids = []
#previous_meshes
#nii_images_list = nii_images_list[0:1]

def process_image(input_folder, image_file, subdivisions, iterations, neta, cell_surfaces, previous_folder, t, subdivide_previous_meshes, output_folder):
    
    if subdivide_previous_meshes:
        previous_meshes = mutils.read_all_meshes(previous_folder)

    #print("Processing time point: ", t, image_file)
    image_path = input_folder + image_file
    cell_meshes = rmg.generate_regular_meshes(image_path, subdivisions, iterations, neta, cell_surfaces, previous_meshes[t-1], subdivide_previous_meshes)
    previous_meshes = cell_meshes

    # Write cell meshes to an OBJ file
    obj_image_file = output_folder  + os.path.splitext(image_file)[0] + '.obj'
    mutils.save_meshes_obj(cell_meshes, t, obj_image_file)
    #t=t+1
    return previous_meshes

def run(i):

    previous_folder = ""
    previous_meshes = []

    for sub in range(2, subdivisions+1):
        if sub == 2:
            t = 1
            previous_folder = output_folder + f"N={sub}/"
            present_folder = output_folder + f"N={sub}/"
            if not os.path.exists(previous_folder): os.makedirs(previous_folder)

            for image_file in nii_images_list:
                image_path = input_folder + image_file
                cell_meshes = rmg.generate_regular_meshes(image_path, sub, iterations, neta, cell_surfaces, previous_meshes, subdivide_previous_meshes=False)
                previous_meshes = cell_meshes

                obj_image_file = previous_folder + os.path.splitext(image_file)[0] + '.obj'
                mutils.save_meshes_obj(cell_meshes, t, obj_image_file)

                t=t+1
        else:    
            t = 1
            #arguments = []
            
            previous_meshes = mutils.read_all_meshes(previous_folder)
            #print(previous_folder , ", " , len(previous_meshes))        

            present_folder = output_folder + f"N={sub}/"
            if not os.path.exists(present_folder): os.makedirs(present_folder)

            #print("number of images: ", len(nii_images_list), batch_length)
            for image_file in nii_images_list:#, batch_length):
                #arguments += [(input_folder, image_file, sub, 50, neta, cell_surfaces, previous_folder, t, True, present_folder)]
                process_image(input_folder, image_file, sub, 50, neta, cell_surfaces, previous_folder, t, True, present_folder)
                t=t+1
            
            previous_folder = present_folder


def run2():
    a = 2+2
    return a

args = [(0),(1)]
num_processes = max(multiprocessing.cpu_count() - 2, 1)
if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=num_processes)
    result = pool.starmap(run, [(0,)])
    pool.close()
    pool.join()
        

#t=t+1
""" if __name__ == '__main__':

    for i in range(0, len(nii_images_list), batch_length):
        pool = multiprocessing.Pool(processes=batch_length)
        if i + batch_length < len(nii_images_list):
            result = pool.starmap(process_image, arguments[i:i+batch_length])
        else:
            result = pool.starmap(process_image, arguments[i:len(nii_images_list)]) #rng = range(i,len(nii_images_list))
        pool.close()
        pool.join() """


# Compute Spherical harmonics representation of the (normalized) radius field on all meshes
"""for i, mesh in enumerate(cell_meshes):
    r, r_norm, theta, phi, dA, _,_,_ = mutils.compute_vertex_properties(mesh)
    f = r_norm
    fi = np.zeros(r.shape)

    frames += [t-1]
    cell_ids += [i]
    Sl += [sh.compute_sphm_inv_rep(f,fi,theta,phi,dA,Lmax)]   
t=t+1 """

""" # Save spherical harmonics representation to csv
columns = [f"S{i}" for i in range(Lmax+1)]
df = pd.DataFrame(Sl)
df.columns = columns
df["fr"] = frames
df["id"] = cell_ids
print(df.head())
df.to_csv(output_folder + '/cells_rot_inv_rep.csv', index=False)
"""