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


def solve_underdetermined_system(A, b):
    
    objective_function = lambda x: np.linalg.norm(np.dot(A, x) - b)**2
    initial_guess = np.zeros(A.shape[1])
    result = minimize(objective_function, initial_guess)

    return result.x, result.fun

def solve_linear_system(A, b):
    print(f"Shape of A: {A.shape}, Shape of b: {b.shape}")
    num_equations, num_unknowns = A.shape

    if num_equations > num_unknowns:
        x, residuals, rank, singular_values = np.linalg.lstsq(A, b, rcond=None)
        return x, np.array(residuals)
    elif num_equations < num_unknowns:
        x, residuals = solve_underdetermined_system(A, b)
        return x
    else:
        x = np.linalg.solve(A, b)
        return x, np.zeros(num_equations)


#root_folder = "obj/" #Put all embryo folders in this directory

start_time_batch = time.time()

# Specify the folder containing input and output folders
input_folder = "../UsefulLogs/3009_all/UsefulLogs_StrainInference3D_717797914938.3593/obj/"
output_folder = input_folder

# List all files in the folder
file_list = os.listdir(input_folder)
obj_images_list = [f for f in sorted(file_list) if f.endswith('.obj')]


active = True
elastic = False
viscous = False


mesh_file = input_folder + obj_images_list[0]
with open(mesh_file, 'r') as obj_file:
    ref_mesh = mutils.read_meshes(obj_file)[0]

#ref_mesh = mutils.read_meshes(obj_images_list[0])#[0]
#print(ref_mesh)
r, r_norm, theta, phi, dA, _, volume, centre = mutils.compute_vertex_properties(ref_mesh)
#Ylm_r, Ylm_i, Ylm = sh.compute_all_harmonics(theta,phi,Lmax)
radius = np.power(volume*0.75/np.pi, 1/3)
ico_sphere = mutils.create_icosphere(radius,centre, 5)
ref_mesh = ico_sphere
nabla_ref = mutils.compute_grad_operator(ref_mesh)

t_max = 3#(len(obj_images_list)-2) #3
obj_images_list = obj_images_list[0:t_max]

columns = ["cid", "id", "fr", "l1", "l2", "l3", "m1", "m2", "m3", "s11", "s12", "s13", "s21", "s22", "s23", "s31", "s32", "s33"]
data = np.zeros((len(r)*(len(obj_images_list)-2), len(columns)))

print(data.shape)
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
    

    d_t = 1
    num_vertices = cell_meshes[0].GetNumberOfPoints()  # Number of equations
    velocities = mutils.compute_velocities(cell_meshes, d_t)
    nabla_t = mutils.compute_grad_operator(cell_meshes[0])
    _, _, _, _, dA, surface_area, _, _ = mutils.compute_vertex_properties(cell_meshes[0])
    
    
    
    E_t = np.zeros((num_vertices, 3,3))
    
    V_t = velocities[0]
    #print(len(velocities))
    V_t_p_1 = velocities[1]
    grad_V_t = np.zeros((num_vertices, 3, 3)) #nabla_t * V_t[:, np.newaxis, :] #(n,n,3) x (n,3)
    for i in range(3):
        for j in range(3):
            grad_V_t[:,i,j] = np.dot(nabla_t[:,:,i], V_t[:,j])
    #grad_V_t = np.einsum('ijk,il->ijlk', nabla_t, V_t) #(n,n,3) x (n,3)
    V_grad_V_t = np.zeros((num_vertices, 3)) #np.einsum('ijk,ik->ij', grad_V_t, V_t)
    for i in range(num_vertices):
        V_grad_V_t[i,:] = np.dot(grad_V_t[i,:,:], V_t[i,:]) 
    
    a_t = (V_t_p_1 - V_t)/d_t + V_grad_V_t
    rho_a_t = np.zeros((num_vertices, 3)) #dA * a_t / surface_area
    for i in range(num_vertices):
        rho_a_t[i, :] = dA[i] * a_t[i,:]/surface_area
    
    
    x_t = mutils.get_vertices(cell_meshes[0])
    F_t = np.zeros((num_vertices, 3, 3)) #np.einsum('ijk,il->ijlk', nabla_ref, x_t) # gradient of x_t with respect to reference mesh
    for i in range(3):
        for j in range(3):
            F_t[:,i,j] = np.dot(nabla_ref[:,:,i], x_t[:,j])
    
    E_t = (np.matmul(np.transpose(F_t, axes=(0, 2, 1)), F_t) - np.eye(3))/2
    D_t = (grad_V_t + np.transpose(grad_V_t, axes=(0, 2, 1)))/2
    E_t_dot = np.matmul(np.matmul(np.transpose(F_t, axes=(0, 2, 1)), D_t), F_t)

    if elastic:
        A_elastic = np.zeros((num_vertices, num_vertices, 3))
        for i in range(3):
            A_elastic[:,:,i] = np.dot(nabla_t[:,:,0], E_t[:,i,0]) + np.dot(nabla_t[:,:,1], E_t[:,i,1]) + np.dot(nabla_t[:,:,2], E_t[:,i,2])
            A_elastic[:,:,i] += (np.transpose(E_t[:,i,0]*nabla_t[:,:,0].T) + np.transpose(E_t[:,i,1]*nabla_t[:,:,1].T) + np.transpose(E_t[:,i,2]*nabla_t[:,:,2].T))
    
    if viscous:
        A_viscous = np.zeros((num_vertices, num_vertices, 3))
        for i in range(3):
            A_viscous[:,:,i] = np.dot(nabla_t[:,:,0], E_t_dot[:,i,0]) + np.dot(nabla_t[:,:,1], E_t_dot[:,i,1]) + np.dot(nabla_t[:,:,2], E_t_dot[:,i,2])
            A_viscous[:,:,i] += (np.transpose(E_t_dot[:,i,0]*nabla_t[:,:,0].T) + np.transpose(E_t_dot[:,i,1]*nabla_t[:,:,1].T) + np.transpose(E_t_dot[:,i,2]*nabla_t[:,:,2].T))
    
    if active:
        A_active = np.zeros((num_vertices, 3*num_vertices, 3))
        for i in range(3):
            A_active[:,:,i] = np.concatenate((nabla_t[:,:,0],nabla_t[:,:,1],nabla_t[:,:,2]), axis=1)

    n_rows = 3 * num_vertices
    n_cols = num_vertices
    
    # Solve Ax=b
    b = rho_a_t
    if active and not viscous and not elastic:
        A = A_active
    if active and not viscous and elastic:
        A = np.concatenate((A_active, A_elastic), axis=1)
    if active and viscous and not elastic:
        A = np.concatenate((A_active, A_viscous), axis=1)
    if active and viscous and elastic:
        A = np.concatenate((A_active, A_viscous, A_elastic), axis=1)
    if not active and not viscous and elastic:
        A = A_elastic
    if not active and viscous and not elastic:
        A = A_viscous
    if not active and viscous and elastic:
        A = np.concatenate((A_viscous, A_elastic), axis=1)

    end_time_file = time.time()
    print(f"Execution time for frame {t} before solving linear system: {end_time_file-start_time_file}")


    if active:
        for i in range(3):
            print(A.shape, b.shape)
            A_T = np.transpose(A[:,:,i])
            x = np.linalg.inv(A_T@A)@A_T@b[:,i]

            res = (A@x-b[:,i])
            res = np.dot(res, res)

            x = np.round(x, 6)

            res = np.round(res, 6)
            print(res, np.sum(x))
            data[start:start+num_vertices, 9 + i*3] = x[0:10242]
            data[start:start+num_vertices, 10 + i*3] = x[10242:20484]
            data[start:start+num_vertices, 11 + i*3] = x[20484:30726]
            data[start:start+num_vertices, 4] = res

    if elastic:
        A = np.concatenate((A[:,:,0],A[:,:,1],A[:,:,2]), axis=0)
        #A_csc = csc_matrix(A)

        print(np.sum(A))

        b = np.concatenate((b[:,0],b[:,1],b[:,2]), axis=0)
        print(f"Shape of A,b: {A.shape},{b.shape}")

        A_T = np.transpose(A)
        x = np.linalg.inv(A_T@A)@A_T@b

        res = (A@x-b)
        res = np.dot(res, res)

        #x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        #x, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var = lsqr(A_csc, b)
        x = np.round(x, 6)

        res = np.round(res, 6)
        #print(x)
        print(res, np.sum(x))
        data[start:start+num_vertices, 3] = x[0:10242]
        #data[start:start+num_vertices, 6] = x[10242:20484]
        data[start:start+num_vertices, 4] = res
    #x = []
    #residuals = []
    #for i in range(3):
    #    y, res = solve_linear_system(A[:,:,i], b[:,i])
    #    x += [y]
    #    residuals += [res]
    
        #data[start:start+num_vertices, i+3] = np.round(x[i], 6)
    data[start:start+num_vertices, 1] = range(0,num_vertices)
    data[start:start+num_vertices, 2] = t * np.ones(num_vertices) #range(start,start+num_vertices)

    #print(np.linalg.norm(residuals[0]), np.linalg.norm(residuals[1]), np.linalg.norm(residuals[2]))
    
    end_time_file = time.time()
    print(f"Execution time for frame {t} after solving linear system: {end_time_file-start_time_file}")

    start += num_vertices

df = pd.DataFrame(columns=columns, data=data)
df.to_csv("../UsefulLogs/3009_all/UsefulLogs_StrainInference3D_717797914938.3593/viscoelastic_params_total_stress.csv", index=False)

end_time_batch = time.time()
print(f"Total execution time: {end_time_batch - start_time_batch:.6f} seconds")