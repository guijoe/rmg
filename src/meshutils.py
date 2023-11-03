import numpy as np
import vtkmodules.all as vtk

# Read mesh from file
def read_mesh(mesh_file):
    reader = vtk.vtkOBJReader()
    reader.SetFileName(mesh_file)
    reader.Update()
    mesh = reader.GetOutput()
    return mesh


def read_meshes(obj_file):
    meshes = []
    current_mesh = None
    max_vertex_index = 1
    
    for line in obj_file:
        line = line.strip()
        
        if line.startswith('g '):
            #print(max_vertex_index)
            if current_mesh: 
                max_vertex_index += len(current_mesh['vertices'])
            
            # Start of a new mesh
            if current_mesh:
                meshes.append(current_mesh)
            current_mesh = {'vertices': [], 'faces': []}
            
            #print(line)

        elif line.startswith('v '):
            # Vertex line
            if current_mesh:
                parts = line.split()
                if len(parts) == 4:
                    # Parse vertex coordinates
                    vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                    current_mesh['vertices'].append(vertex)
                    

        elif line.startswith('f '):
            # Face line
            if current_mesh:
                parts = line.split()
                if len(parts) >= 4:
                    # Parse face indices, offset by the max_vertex_index
                    face_indices = [int(part.split('/')[0]) - max_vertex_index for part in parts[1:]]
                    current_mesh['faces'].append(face_indices)
        
        #max_vertex_index += 1

    if current_mesh:
        meshes.append(current_mesh)

    meshes = create_vtk_mesh(meshes)
    #for i, mesh in enumerate(meshes):
    #    print(f"Mesh {i + 1}:")
    #    print(f"Vertices: {mesh.GetNumberOfPoints()}")
    #    print(f"Faces: {mesh.GetNumberOfCells()}")
    #    print()

    return meshes


def create_vtk_mesh(mesh_data):
    vtk_meshes = []

    for mesh in mesh_data:
        # Create VTK points from vertices
        points = vtk.vtkPoints()
        for vertex in mesh['vertices']:
            points.InsertNextPoint(*vertex)

        # Create VTK cells from faces
        cells = vtk.vtkCellArray()
        for face in mesh['faces']:
            if len(face) == 3:
                cells.InsertNextCell(3, face)  # Assume triangular faces

        # Create VTK polydata and set points and cells
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetPolys(cells)

        vtk_meshes.append(polydata)

    return vtk_meshes

""" # Open and read your OBJ file
with open('your_file.obj', 'r') as obj_file:
    meshes = parse_obj_file(obj_file)

# Now, `meshes` is a list where each element is a dictionary representing a mesh.
# Each mesh has a 'vertices' key with a list of vertices and a 'faces' key with a list of faces.

for i, mesh in enumerate(meshes):
    print(f"Mesh {i + 1}:")
    print(f"Vertices: {len(mesh['vertices'])}")
    print(f"Faces: {len(mesh['faces'])}")
    print() """


# Compute all the triangles of a vertex
def get_vertex_triangles(mesh):
    num_vertices = mesh.GetNumberOfPoints()

    triangles_per_vertex = [[] for _ in range(num_vertices)]
    triangles = []

    for i in range(mesh.GetNumberOfCells()):
        triangle = mesh.GetCell(i)
        vertices = [triangle.GetPointId(j) for j in range(3)]
        #print(np.min(vertices), np.max(vertices))

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

    mass_properties = vtk.vtkMassProperties()
    mass_properties.SetInputData(mesh)
    mass_properties.Update()

    volume = mass_properties.GetVolume()
    surface_area = mass_properties.GetSurfaceArea()
    r_norm = r / np.power(volume, 1/3)

    # dA
    dA = np.zeros(num_vertices)
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
                K_val = np.sqrt(S * (S - xixj_norm) * (S - xjxk_norm) * (S - xkxi_norm))
                
                radius = xixj_norm * xjxk_norm * xkxi_norm / (4 * K_val)
                val = np.sqrt(np.abs(radius * radius - xixj_norm*xixj_norm / 4))

                side1 = val * IJNormalUnit
                side2 = val * IKNormalUnit

                dA[i] += (xixj_norm * np.linalg.norm(side1) + xkxi_norm * np.linalg.norm(side2)) / 4
    #print(sum(dA), surface_area)       
    return r, r_norm, theta, phi, dA, surface_area

def compute_gradient_div(mesh, field):
    
    num_vertices = mesh.GetNumberOfPoints()
    num_triangles = mesh.GetNumberOfCells()
    vertices = np.array([np.array(mesh.GetPoint(i)) for i in range(num_vertices)])

    div = np.zeros(field.shape)

    gradient = np.zeros(div.shape + (3,))
    
    for n in range(num_triangles):

        tri_gradient = np.zeros(gradient[0].shape)
        
        cell = mesh.GetCell(n)

        i = cell.GetPointId(0)
        j = cell.GetPointId(1)
        k = cell.GetPointId(2)

        fi = field[i]
        fj = field[j]
        fk = field[k]

        # Calculate edge vectors
        xixj = vertices[j] - vertices[i]
        xixk = vertices[k] - vertices[i]
        
        ijk_normal = np.cross(xixj, xixk)
        ij_normal = np.cross(ijk_normal, xixj)
        ij_normal = np.linalg.norm(xixj) * ij_normal / np.linalg.norm(ij_normal)
        ik_normal = np.cross(ijk_normal, xixk)
        ik_normal = np.linalg.norm(xixk) * ik_normal / np.linalg.norm(ik_normal)

        tri_gradient = ((fj - fi) * ij_normal + (fk - fi) * ik_normal)/np.linalg.norm(ijk_normal)

        gradient[i] += tri_gradient
        gradient[j] += tri_gradient
        gradient[k] += tri_gradient

    div = np.sum(gradient, axis=gradient.shape[-1]-1)
    
    return gradient, div