import os
import sys
import time
import numpy as np
import pandas as pd
import src.rmg as rmg
import src.sphm as sh
#from mpi4py import MPI
import src.meshutils as mutils
from scipy.optimize import minimize
from scipy.sparse.linalg import lsqr
from scipy.sparse import csc_matrix


# Function to compute eigenvalues and eigenvectors for each matrix in the tensor
def compute_eigenvalues_eigenvectors(tensor):
    eigenvalues_list = []
    eigenvectors_list = []

    for i in range(tensor.shape[0]):
        matrix = tensor[i, :, :]
        eigenvalues, eigenvectors = np.linalg.eig(matrix)

        eigenvalues_list.append(eigenvalues)
        eigenvectors_list.append(eigenvectors)

    return eigenvalues_list, eigenvectors_list

#root_folder = "obj/" #Put all embryo folders in this directory

start_time_batch = time.time()

# Specify the folder containing input and output folders
#input_folder = "../UsefulLogs/3009_all/UsefulLogs_StrainInference3D_717797914938.3593/obj/"
input_folder = "/Users/joeld/Downloads/Astec-Pm1_old/emb/N=2/"
output_folder = input_folder

# List all files in the folder
file_list = os.listdir(input_folder)
obj_images_list = [f for f in sorted(file_list) if f.endswith('.obj')]


#active = True
#elastic = False
#viscous = False


mesh_file = input_folder + obj_images_list[0]
with open(mesh_file, 'r') as obj_file:
    ref_mesh = mutils.read_meshes(obj_file)[0]

#ref_mesh = mutils.read_meshes(obj_images_list[0])#[0]
#print(ref_mesh)
r, r_norm, theta, phi, dA, _, volume, centre = mutils.compute_vertex_properties(ref_mesh)
#Ylm_r, Ylm_i, Ylm = sh.compute_all_harmonics(theta,phi,Lmax)
radius = np.power(volume*0.75/np.pi, 1/3)
ico_sphere = mutils.create_icosphere(radius,centre, 2)
ref_mesh = ico_sphere


#nabla_ref = mutils.compute_grad_operator(ref_mesh)

r2, r_norm, theta2, phi2, dA2, _, volume, centre = mutils.compute_vertex_properties(ref_mesh)
#print(np.min(theta), np.max(theta),np.min(phi), np.max(phi),np.min(dA), np.max(dA), np.min(r),np.max(r))

t_max = 3#(len(obj_images_list)-2) #3
obj_images_list = obj_images_list[0:t_max]

columns = ["cid", "id", "fr", "Dxx", "Dxy", "Dxz", "Dyx", "Dyy", "Dyz", "Dzx", "Dzy", "Dzz", "Dl1", "Dl2", "Dl3", "e1x", "e1y", "e1z", "e2x", "e2y", "e2z", "e3x", "e3y", "e3z", "x", "y", "z", "Tta2", "Phi2", "vA2","r2","Tta", "Phi", "vA","r"]
data = np.zeros((len(r)*(len(obj_images_list)-2), len(columns)))

#print("data shape: ", data.shape)
start = 0
for t in range(len(obj_images_list)-2):
    start_time_file = time.time()

    #image_file = obj_images_list[t0]
    mesh_tau = []
    cell_meshes = []
    for tau in range(t, t+3):
        mesh_file = input_folder + obj_images_list[tau]
        with open(mesh_file, 'r') as obj_file:
            cell_meshes += [mutils.read_meshes(obj_file)[0]]
        #print(mesh_file)
    
     
    d_t = 1
    num_vertices = cell_meshes[0].GetNumberOfPoints()  # Number of equations
    velocities = mutils.compute_velocities(cell_meshes, d_t)
    nabla_t = mutils.compute_grad_operator(cell_meshes[0])
    r, _, theta, phi, dA, surface_area, _, _ = mutils.compute_vertex_properties(cell_meshes[0])
    
    
    #E_t = np.zeros((num_vertices, 3,3))
    x_t = mutils.get_vertices(cell_meshes[0])
    
    V_t = velocities[0]
    #print(len(velocities))
    #V_t_p_1 = velocities[1]
    grad_V_t = np.zeros((num_vertices, 3, 3)) #nabla_t * V_t[:, np.newaxis, :] #(n,n,3) x (n,3)
    for i in range(3):
        for j in range(3):
            grad_V_t[:,i,j] = np.dot(nabla_t[:,:,i], V_t[:,j])

    grad_V_t_T = np.zeros((num_vertices, 3, 3))
    for n in range(grad_V_t.shape[0]):
        grad_V_t_T[n,:,:] = grad_V_t[n,:,:].T
    
    D_t = (grad_V_t + grad_V_t_T)/2
    
    tD_t, nD_t = mutils.tn_decompose_tensor(cell_meshes[0], D_t)
    #print("tangential: ", tD_t)
    #print("normal: ", nD_t)

    # Compute eigenvalues and eigenvectors
    eigen_values, eigen_vectors = compute_eigenvalues_eigenvectors(D_t)
    #print(eigen_values)
    
    data[start:start+num_vertices, 1] = range(0,num_vertices)
    data[start:start+num_vertices, 2] = t * np.ones(num_vertices)
    data[start:start+num_vertices, 3:12] = np.reshape(D_t,(D_t.shape[0], 9))
    data[start:start+num_vertices, 12:15] = eigen_values
    data[start:start+num_vertices, 15:24] = np.reshape(eigen_vectors,(len(eigen_vectors), 9))
    data[start:start+num_vertices, 24:27] = x_t
    data[start:start+num_vertices, 27] = theta2
    data[start:start+num_vertices, 28] = phi2
    data[start:start+num_vertices, 29] = dA2
    data[start:start+num_vertices, 30] = r2
    data[start:start+num_vertices, 31] = theta
    data[start:start+num_vertices, 32] = phi
    data[start:start+num_vertices, 33] = dA
    data[start:start+num_vertices, 34] = r
    start += num_vertices

    end_time_file = time.time()
    print(f"Execution time for frame {t}/{len(obj_images_list)-2}: {end_time_file - start_time_file:.6f} seconds")

df = pd.DataFrame(columns=columns, data=np.round(data,6))
#df.to_csv("../UsefulLogs/3009_all/UsefulLogs_StrainInference3D_717797914938.3593/vertexStats_python1.csv", index=False)

end_time_batch = time.time()
print(f"Total execution time: {end_time_batch - start_time_batch:.6f} seconds") #"""