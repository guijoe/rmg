import numpy as np
import vtkmodules.all as vtk

# Read mesh from file
def read_mesh(mesh_file):
    reader = vtk.vtkOBJReader()
    reader.SetFileName(mesh_file)
    reader.Update()
    mesh = reader.GetOutput()
    return mesh

# Compute all the triangles of a vertex
def get_vertex_triangles(mesh):
    num_vertices = mesh.GetNumberOfPoints()

    triangles_per_vertex = [[] for _ in range(num_vertices)]
    triangles = []

    for i in range(mesh.GetNumberOfCells()):
        triangle = mesh.GetCell(i)
        vertices = [triangle.GetPointId(j) for j in range(3)]

        triangles += [vertices] 
        for vertex in vertices:
            triangles_per_vertex[vertex] += [i]#.append(i)
    return triangles_per_vertex, triangles

# Compute vertex properties: dA, r, theta, phi
def compute_vertex_properties(mesh):
    num_vertices = mesh.GetNumberOfPoints()
    triangles_per_vertex, triangles = get_vertex_triangles(mesh)

    vertices = np.zeros((num_vertices, 3))
    for i in range(num_vertices):
        vertices[i,:] = mesh.GetPoint(i)
    center = np.mean(vertices, axis=0)
    centered_vertices = vertices - center
    
    # r, theta, phi
    r = np.linalg.norm(centered_vertices, axis=1)
    phi = np.arccos(centered_vertices[:, 2] / r)
    theta = np.arctan2(centered_vertices[:, 1], centered_vertices[:, 0]) + np.pi

    dA = np.zeros(num_vertices)
    K = np.zeros((num_vertices, 3))
    for i in range(num_vertices):
        xi = i
        for j in range(len(triangles_per_vertex[i])):
            tri = triangles[triangles_per_vertex[i][j]]
            
            xj = tri[1] if xi == tri[0] else tri[2] if xi == tri[1] else tri[0]
            xk = tri[2] if xi == tri[0] else tri[0] if xi == tri[1] else tri[1]
            
            # Compute Areas
            
            xixj = vertices[xj] - vertices[xi]
            xjxk = vertices[xk] - vertices[xj]
            xkxi = vertices[xi] - vertices[xk]
            xixj_norm = np.linalg.norm(xixj)
            xjxk_norm = np.linalg.norm(xjxk)
            xkxi_norm = np.linalg.norm(xkxi) 

            tri_area = np.linalg.norm(np.cross(xixj,xjxk))/2
            #print(i,j,tri_area, xixj_norm, xjxk_norm, xkxi_norm)
            if xjxk_norm*xjxk_norm > xixj_norm*xixj_norm + xkxi_norm*xkxi_norm:
                dA[i] += tri_area/2
            elif xixj_norm*xixj_norm > xjxk_norm*xjxk_norm + xkxi_norm*xkxi_norm:
                dA[i] += tri_area/4
            elif xkxi_norm*xkxi_norm > xixj_norm*xixj_norm + xjxk_norm*xjxk_norm:
                dA[i] += tri_area/4
            else:
                # Compute Voronoi Area
                IJNormalUnit = np.cross(xixj,np.cross(xixj,xkxi))/np.linalg.norm(np.cross(xixj,np.cross(xixj,xkxi)))
                IKNormalUnit = np.cross(xixj,np.cross(xixj,xkxi))/np.linalg.norm(np.cross(xkxi,np.cross(xixj,xkxi)))
                
                S = (xixj_norm + xjxk_norm + xkxi_norm) / 2
                K_val = np.sqrt(S * (S - xkxi_norm) * (S - xjxk_norm) * (S - xkxi_norm))
                
                radius = xkxi_norm * xjxk_norm * xkxi_norm / (4 * K_val)
                val = np.sqrt(np.abs(radius * radius - xixj_norm*xixj_norm / 4))

                side1 = val * IJNormalUnit
                side2 = val * IKNormalUnit

                dA[i] += (xixj_norm * np.linalg.norm(side1) + xkxi_norm * np.linalg.norm(side2)) / 4
            
    return r, theta, phi, dA
