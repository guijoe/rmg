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
input_folder = "obj/Karim/"
output_folder = input_folder

# List all files in the folder
file_list = os.listdir(input_folder + "obj")
obj_images_list = [f for f in sorted(file_list) if f.endswith('.obj')]

Nmax = 3 # Maximum polynomial evolution law
Tmax = 10 # Number of simulation steps
Lmax = 20 # Maximum spherical harmonic
Smax = 20 # Number of initial sample states
subset = 10 * np.arange(Smax) # Starting time of different sample states

print(subset)


frames = []
cell_ids = []
for t0 in subset:
    start_time_file = time.time()
    
    image_file = obj_images_list[t0]
    image_path = input_folder + "obj/" + image_file

    with open(image_path, 'r') as obj_file:
        cell_meshes = mutils.read_meshes(obj_file)
    

    # Compute Spherical harmonics
    for i, mesh in enumerate(cell_meshes):
        start_time_mesh = time.time()

        r, r_norm, theta, phi, dA, _, volume, centre = mutils.compute_vertex_properties(mesh)
        Ylm_r, Ylm_i, Ylm = sh.compute_all_harmonics(theta,phi,Lmax)

        radius = np.power(volume*0.75/np.pi, 1/3)
        icosphere = mutils.create_icosphere(radius,centre, 5)

        F0 = mutils.grad_change(icosphere,mesh)
        E0 = np.dot(F0.transpose(), F0) - np.eye(3)

        

        for l in range(Lmax):
            for m in range(2*l+1):
                print("hello")
        
        # Write cell meshes to an OBJ file
        obj_image_file = output_folder + "obj/" + os.path.splitext(image_file)[0] + '.obj'
        rmg.save_meshes_obj(cell_meshes, t, obj_image_file)

        end_time_mesh = time.time()
        #print(f"Mesh {i}/{len(cell_meshes)} of time {t}/{len(obj_images_list)} execution time: {end_time_mesh - start_time_mesh:.6f} seconds")  
    t=t+1

    end_time_file = time.time()
    #print(f"File {t}/{len(obj_images_list)} execution time: {end_time_file - start_time_file:.6f} seconds")

# Save spherical harmonics representation to csv
columns = [f"S{i}" for i in range(Lmax+1)]
df = pd.DataFrame(Sl)
df.columns = columns
df["fr"] = frames
df["id"] = cell_ids
#print(df.head())
df.to_csv(output_folder + '/cells_rot_inv_rep.csv', index=False)

end_time_batch = time.time()
print(f"Total execution time: {end_time_batch - start_time_batch:.6f} seconds")