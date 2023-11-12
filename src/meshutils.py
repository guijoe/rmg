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

        #print("hello" , line)
        
        if line.startswith('g '):
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

    #print(meshes)
    meshes = create_vtk_mesh(meshes)
    #for i, mesh in enumerate(meshes):
    #    print(f"Mesh {i + 1}:")
    #    print(f"Vertices: {mesh.GetNumberOfPoints()}")
    #    print(f"Faces: {mesh.GetNumberOfCells()}")
    #    print()

    return meshes

# Creates a basic icosahedron sphere of 12 vertices
def create_base_ico_sphere():
    
    # Create vertices
    theta = 26.56505117707799 * np.pi / 180
    stheta = np.sin(theta)
    ctheta = np.cos(theta)

    vertices = []
    vertices += [(0.0, 0.0, -1.0)]

    phi = np.pi / 5
    for i in range(1,6): 
        vertices +=[(ctheta * np.cos(phi), ctheta * np.sin(phi), -stheta)]
        phi += 2 * np.pi / 5

    phi = 0
    for i in range(6,11):
        vertices +=[(ctheta * np.cos(phi), ctheta * np.sin(phi), stheta)]
        phi += 2 * np.pi / 5

    vertices += [(0, 0, 1)]

    # Normalize the vertices to form a unit sphere
    vertices = [tuple(np.array(v) / np.linalg.norm(v)) for v in vertices]

    points = vtk.vtkPoints()
    # Add vertices to the mesh
    for v in vertices:
        points.InsertNextPoint(v)

    # Create triangles
    icosahedron_faces = [
            (0, 2, 1),
            (0, 3, 2),
            (0, 4, 3),
            (0, 5, 4),
            (0, 1, 5),
            (1, 2, 7),
            (2, 3, 8),
            (3, 4, 9),
            (4, 5, 10),
            (5, 1, 6),
            (1, 7, 6),
            (2, 8, 7),
            (3, 9, 8),
            (4, 10, 9),
            (5, 6, 10),
            (6, 7, 11),
            (7, 8, 11),
            (8, 9, 11),
            (9, 10, 11),
            (10, 6, 11),
        ]

    triangles = vtk.vtkCellArray()
    for i, tri in enumerate(icosahedron_faces):
        triangles.InsertNextCell(3)
        for j in tri:
            triangles.InsertCellPoint(j)

    # Create a polydata object
    mesh = vtk.vtkPolyData()
    mesh.SetPoints(points)
    mesh.SetPolys(triangles)

    return mesh

# Creates a sphere given a radius, centre, based of the subdivisions of the basic icosahedron sphere
def create_icosphere(radius: float, centre, subdivisions: int):
    icosphere = create_base_ico_sphere()

    # Loop subdivision
    subdivide = vtk.vtkLoopSubdivisionFilter()
    subdivide.SetNumberOfSubdivisions(subdivisions)
    subdivide.SetInputData(icosphere)
    subdivide.Update()

    icosphere = subdivide.GetOutput()

    points = icosphere.GetPoints()
    vertices = [np.array(points.GetPoint(i)) for i in range(points.GetNumberOfPoints())]
    vertices = [tuple(np.array(v) / np.linalg.norm(v)) for v in vertices]
    vertices = [np.array(points.GetPoint(i))*radius for i in range(points.GetNumberOfPoints())]
    vertices = [np.array(v) + np.array(centre) for v in vertices]
    
    for i in range(points.GetNumberOfPoints()):
        points.SetPoint(i, vertices[i])
    icosphere.SetPoints(points)
    
    icosphere.Modified()
    return icosphere

def read_all_meshes(obj_folder, multiple):
    cell_meshes = []

    for obj_file in obj_folder:
        obj_path = obj_folder + obj_file

        meshes = read_meshes(obj_path)

    if multiple:
        return cell_meshes
    else:
        return cell_meshes[0]
        
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

def get_vertex_1_2_ring(mesh):
    num_vertices = mesh.GetNumberOfPoints()
    tris_per_vertex, tris = get_vertex_triangles(mesh)

    ring1 = []
    ring2 = []

    #ring1 = [[k for k in tris[tris_per_vertex[i][j]] if k != i] for j in ]

    for i in range(num_vertices):
        ring1 += [[]]
        
        for j in range(len(tris_per_vertex[i])):
            tri = tris[tris_per_vertex[i][j]]

            if tri[0] != i: ring1[i] += [tri[0]]
            if tri[1] != i: ring1[i] += [tri[1]]
            if tri[2] != i: ring1[i] += [tri[2]]

        ring1[i] = list(set(ring1))

    for i in range(num_vertices):
        ring2 += [[]]

        for j in ring1[i]:
            ring2[i] += [k for k in ring1[j] if k not in ring1[i]]

    return ring1, ring2


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
    r_norm = r / np.power(0.75 * volume/np.pi, 1/3)

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
    #print("results: ", center)#, phi, dA, surface_area, volume, center)       
    return r, r_norm, theta, phi, dA, surface_area, volume, center

def compute_gradient_div(mesh, field):
    
    num_vertices = mesh.GetNumberOfPoints()
    num_triangles = mesh.GetNumberOfCells()
    vertices = np.array([np.array(mesh.GetPoint(i)) for i in range(num_vertices)])

    div = np.zeros(field.shape)
    total_area_per_vertex = np.zeros(field.shape[0])
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
        ijk_area = 0.5 * np.linalg.norm(ijk_normal)
        ij_normal = np.cross(ijk_normal, xixj)
        ij_normal = np.linalg.norm(xixj) * ij_normal / np.linalg.norm(ij_normal)
        ik_normal = np.cross(ijk_normal, xixk)
        ik_normal = np.linalg.norm(xixk) * ik_normal / np.linalg.norm(ik_normal)

        tri_gradient = ((fj - fi) * ij_normal + (fk - fi) * ik_normal)/np.linalg.norm(ijk_normal)

        gradient[i] += tri_gradient * ijk_area
        gradient[j] += tri_gradient * ijk_area
        gradient[k] += tri_gradient * ijk_area

        total_area_per_vertex[i] += ijk_area
        total_area_per_vertex[j] += ijk_area
        total_area_per_vertex[k] += ijk_area 
    
    for i in range(num_vertices):
        gradient[i,:,:] /= total_area_per_vertex[i]

    div = np.sum(gradient, axis=gradient.shape[-1]-1)
    
    return gradient, div

def grad_change(mesh1, mesh2):
    change_field = np.array([np.array(mesh2.GetPoints(i) - mesh1.GetPoints(i)) for i in mesh1.GetNumberOfPoints()])

    gradient, _ = compute_gradient_div(mesh1, change_field)

    return gradient

def compute_velocities(meshes, d_t):
    velocities = []
    for i in range(len(meshes) - 1):
        vel = np.array([np.array(meshes[i+1].GetPoint(j)) - np.array(meshes[i].GetPoint(j)) for j in range(meshes[i].GetNumberOfPoints())])
        velocities += [vel / d_t]
    return velocities

def compute_normals(mesh):

    num_vertices = mesh.GetNumberOfPoints()

    #### Compute normals and mean curvatures at every mesh vertex
    normals_filter = vtk.vtkPolyDataNormals()
    normals_filter.SetInputData(mesh)
    normals_filter.ComputePointNormalsOn()
    normals_filter.ComputeCellNormalsOff()
    normals_filter.Update()
    normals = [normals_filter.GetOutput().GetPointData().GetNormals().GetTuple(i) for i in range(num_vertices)]
    
    return normals

def compute_grad_operator(mesh):
    num_vertices = mesh.GetNumberOfPoints()
    triangles_per_vertex, triangles = get_vertex_triangles(mesh)
    vertices = np.array([np.array(mesh.GetPoint(i)) for i in range(num_vertices)])

    #normals = compute_normals(mesh)
    
    nabla = np.zeros((num_vertices, num_vertices, 3))
    for i in range(num_vertices):
        xi = i
        S_i = 0
        Ai = np.zeros((len(triangles_per_vertex[i]), 3))

        for j in range(len(triangles_per_vertex[i])):
            tri = triangles[triangles_per_vertex[i][j]]
            
            xj = tri[1] if xi == tri[0] else tri[2] if xi == tri[1] else tri[0]
            xk = tri[2] if xi == tri[0] else tri[0] if xi == tri[1] else tri[1]

            xixj = vertices[xj] - vertices[xi]
            """ xixk = vertices[xk] - vertices[xi]
            #if i==0:
            #    print(vertices[xi],vertices[xj],vertices[xk])
            
            nijk = np.cross(xixj, xixk)
            #nijk_normed = nijk / np.linalg.norm(nijk)
            nij = np.cross(nijk, xixj)
            nij = np.linalg.norm(xixj) * nij / np.linalg.norm(nij)
            nik = np.cross(nijk, -xixk)
            nik = np.linalg.norm(xixk) * nik / np.linalg.norm(nik)
            S_i += np.linalg.norm(nijk)/2
            #if i==1:
            #    print(nij, nik, nijk, xi, xj, xk)
            #if(xixj[0]*xixj[1]*xixj[2] != 0):
            nabla[xi,xi,:] -= (nij+nik)
            nabla[xi,xj,:] += nik
            nabla[xi,xk,:] += nij """ 

            Ai[j,:] = -xixj
        #nabla[xi,:,:] /= S_i

        Ai_T = np.transpose(Ai)
        Gi = np.linalg.inv(Ai_T @ Ai) @ Ai_T
        
        for j in range(len(triangles_per_vertex[i])):
            
            tri = triangles[triangles_per_vertex[i][j]]
            xj = tri[1] if xi == tri[0] else tri[2] if xi == tri[1] else tri[0]
            
            nabla[xi,xj,:] = Gi[:,j]
            nabla[xi,xi,:] -= Gi[:,j]

        #if i==0:
        #    print(Gi, Gi[:,0])

    return nabla

def get_vertices(mesh):
    num_vertices = mesh.GetNumberOfPoints()
    vertices = np.array([np.array(mesh.GetPoint(i)) for i in range(num_vertices)])

    return vertices

#def compute_div_operator(mesh):
#    nabla = compute_grad_operator(mesh)
