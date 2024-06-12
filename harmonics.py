import os
import sys
import time
import numpy as np
import pandas as pd
import src.rmg as rmg
import src.sphm as sh
import src.meshutils as mutils

#root_folder = "obj/" #Put all embryo folders in this directory

start_time_batch = time.time()

# Specify the folder containing input and output folders
#input_folder = "obj/Karim/"
#input_folder = "obj/Astec/"

input_folder = sys.argv[1]
output_folder = input_folder

# List all files in the folder
file_list = os.listdir(input_folder + "obj")
obj_images_list = [f for f in sorted(file_list) if f.endswith('.obj')]
#print(obj_images_list)

t = 1
Lmax = 20
Sl = []
Pl = []
frames = []
cell_ids = []

unit_sphere = mutils.create_icosphere(1,(0,0,0),2)
_, _, theta, phi, dA, surface_area, _, _= mutils.compute_vertex_properties(unit_sphere)
Ylm_r, Ylm_i,Ylm=sh.compute_all_harmonics(theta,phi,Lmax)

#obj_images_list = obj_images_list[0:9]
for image_file in obj_images_list:
    
    start_time_file = time.time()
    
    image_path = input_folder + "obj/" + image_file
    with open(image_path, 'r') as obj_file:
        cell_meshes = mutils.read_meshes(obj_file)
    #cell_meshes = mutils.read_meshes(image_path)

    #print(len(cell_meshes))

    # Compute Spherical harmonics representation of the radius field on all meshes
    #cell_meshes = cell_meshes[0:1]
    for i, mesh in enumerate(cell_meshes):
        start_time_mesh = time.time()
        _, r_norm, _, _, _, _, _, _= mutils.compute_vertex_properties(mesh)
        f = r_norm
        fi = np.zeros(r_norm.shape)

        frames += [t-1]
        cell_ids += [i]
        #sh1 = sh.compute_sphm_inv_rep(f,fi,theta,phi,dA,Lmax)
        #print(sh1)
        #Pl += [sh.compute_sphm_inv_rep4(f,fi,theta,phi,dA,Lmax)]
        Pl += [sh.compute_sphm_inv_rep2(f, fi, Ylm_r, Ylm_i, Ylm, dA, Lmax)]
        #print(Pl)
        #Sl += [sh.compute_sphm_inv_rep3(f,fi,theta,phi,dA,Lmax)]
        Sl += [sh.compute_sphm_inv_rep(f, fi, Ylm_r, Ylm_i, Ylm, dA, Lmax)]

        end_time_mesh = time.time()
        print(f"Mesh {i}/{len(cell_meshes)} of time {t}/{len(obj_images_list)} execution time: {end_time_mesh - start_time_mesh:.6f} seconds")  
    t=t+1

    end_time_file = time.time()
    #print(f"File {t}/{len(obj_images_list)} execution time: {end_time_file - start_time_file:.6f} seconds")

# Save spherical harmonics representation to csv
S_cols = [f"S{i}" for i in range(Lmax+1)]
P_cols = [f"P{i}" for i in range(Lmax+1)]

columns = S_cols + P_cols

data = np.concatenate((Sl,Pl),axis=1)
df = pd.DataFrame(data)
df.columns = columns
df["fr"] = frames
df["id"] = cell_ids
#print(df.head())
df.to_csv(output_folder + '/cells_rot_inv_rep.csv', index=False)

end_time_batch = time.time()
print(f"Total execution time: {end_time_batch - start_time_batch:.6f} seconds")