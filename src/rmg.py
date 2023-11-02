import time
import morphonet
import numpy as np
import vtkmodules.all as vtk
from scipy.spatial import KDTree
from skimage import morphology,measure


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

def create_icosphere2(radius: float, centre, subdivisions: int, previous_mesh: vtk.vtkPolyData()):
    icosphere = create_icosphere(radius, centre, subdivisions)

    points1 = icosphere.GetPoints()
    points2 = previous_mesh.GetPoints()
    vertices = [(np.array(points1.GetPoint(i)) + np.array(points2.GetPoint(i)))/2 for i in range(points1.GetNumberOfPoints())]
    
    for i in range(len(vertices)):
        points1.SetPoint(i, vertices[i])
    icosphere.SetPoints(points1)
    
    icosphere.Modified()
    return icosphere
    
    return icosphere

# Computes the minimum distance to the pixel cloud and the indices of the closest pixels
def min_distance_index(vertices, surface_pixels):
    min_distance = np.zeros(len(vertices))
    min_index = np.zeros(len(vertices))
    for i in range(len(vertices)):
        distances_from_i = np.sqrt(np.sum((surface_pixels - vertices[i])**2, axis=1))
        min_distance[i] = np.min(distances_from_i)
        min_index[i] = np.argmin(distances_from_i)
    return min_distance,min_index

def min_distance_index2(vertices, surface_pixels_per_vertex, nearest_indices):
    min_distance = np.zeros(len(vertices))
    min_index = np.zeros(len(vertices))
    for i in range(len(vertices)):
        distances_from_i = np.sqrt(np.sum((surface_pixels_per_vertex[i] - vertices[i])**2, axis=1))
        min_distance[i] = np.min(distances_from_i)
        min_index[i] = nearest_indices[i][int(np.argmin(distances_from_i))]
    return min_distance,min_index

# Saves mesh in obj format
def save_meshes_obj(meshes, time, objfile):
    tri_start = 1
    meshes_str = ""
    for i in range(0, len(meshes)):
        mesh_str = "g " + str(time) + "," + str(i) + ",0\n"

        for j in range(0, meshes[i].GetNumberOfPoints()):
            p = np.round(np.array(meshes[i].GetPoint(j)), 6)
            mesh_str += "v " + str(p[0]) + " " + str(p[1]) + " " + str(p[2]) + "\n"

        for j in range(0, meshes[i].GetNumberOfCells()):
            tri = meshes[i].GetCell(j)
            p_ids = tri.GetPointIds()
            mesh_str += "f " + str(tri_start + p_ids.GetId(0)) + " " + str(tri_start + p_ids.GetId(1)) + " " + str(tri_start + p_ids.GetId(2)) + "\n"

        meshes_str += mesh_str
        tri_start += meshes[i].GetNumberOfPoints()

    with open(objfile, "w") as file:
        file.write(meshes_str)

def generate_regular_meshes(image_file, subdivisions, iterations, neta, cell_surfaces, previous_meshes):

    image = morphonet.tools.imread(image_file)
    labeled_image = measure.label(image)
    regions = measure.regionprops(labeled_image)

    cell_meshes = []
    nb_regions = len(regions)+1
    regions_range = range(2, nb_regions) if cell_surfaces else range(1,2)
    
    # Iterate through the labeled regions (excluding background)
    for cl in regions_range:#nb_regions):
        start_time_cell = time.time()
    
        #### Comupte surface pixels
        cell_mask = labeled_image == cl
        cell_surface_mask = cell_mask ^ morphology.erosion(cell_mask, morphology.cube(3)) if cell_surfaces else cell_mask ^ morphology.erosion(cell_mask)
        s_p = np.where(cell_surface_mask)
        surface_pixels = np.array([(s_p[0][i], s_p[1][i], s_p[2][i]) for i in range(len(s_p[0]))])
        
        #a_p = np.where(cell_surface_mask)
        #all_pixels = np.array([(a_p[0][i], a_p[1][i], a_p[2][i]) for i in range(len(a_p[0]))])

        #### Compute the tree of neighbours
        kdtree = KDTree(surface_pixels)
        neighborhood_radius = 3.0
        pixel_neighbours = [kdtree.query_ball_point(p, r=neighborhood_radius) for p in surface_pixels]
        
        #### Compute a spherical mesh surrounding the cell
        centre = np.mean(surface_pixels, axis=0)
        max_radius = 1.25*np.max(np.sqrt(np.sum((surface_pixels - centre)**2, axis=1)))
        icosphere = create_icosphere(max_radius,centre, subdivisions)
       
        # Initialise with the previous state of the cell // probably need to pass the previous computed meshes as parameters
        # Works well with embryo surfaces, for cellular surfaces, the lineage is needed. This has not been implemented yet in this case
        if not cell_surfaces and len(previous_meshes) > 0:
            icosphere = create_icosphere2(max_radius, centre, subdivisions, previous_meshes[cl-1]) 
        
        points = icosphere.GetPoints()
        vertices = [points.GetPoint(i) for i in range(points.GetNumberOfPoints())]
        distances, closest_pixel_indices = min_distance_index(vertices, surface_pixels)
        closest_pixel_indices = closest_pixel_indices.astype(int)

        #### Iterate the update of the vertex positions until the mesh takes the shape of the cell
        for epoch in range(0, iterations):
            
            #### Compute the distances of mesh vertices to the surface of the cell
            nearest_indices = [pixel_neighbours[closest_pixel_indices[i]] for i in range(len(vertices))]
            surface_pixels_per_vertex = [surface_pixels[pixel_neighbours[closest_pixel_indices[i]]] for i in range(len(vertices))]
            distances, closest_pixel_indices = min_distance_index2(vertices, surface_pixels_per_vertex, nearest_indices)
            closest_pixel_indices = closest_pixel_indices.astype(int)
            
            #### Compute distance gradient for all mesh vertices in the 3d space
            dl = 1 # step
            
            # grad x
            vertices_x = [np.array(p) + [dl, 0, 0] for p in vertices]
            distances_x = [np.min(np.sqrt(np.sum((surface_pixels[pixel_neighbours[closest_pixel_indices[i]]] - vertices_x[i])**2, axis=1))) for i in range(len(vertices_x))]
            grad_x = np.array(distances_x) - np.array(distances)

            # grad y
            vertices_y = [np.array(p) + [0, dl, 0] for p in vertices]
            distances_y = [np.min(np.sqrt(np.sum((surface_pixels[pixel_neighbours[closest_pixel_indices[i]]] - vertices_y[i])**2, axis=1))) for i in range(len(vertices_y))]
            grad_y = np.array(distances_y) - np.array(distances)

            # grad z
            vertices_z = [np.array(p) + [0, 0, dl] for p in vertices]
            distances_z = [np.min(np.sqrt(np.sum((surface_pixels[pixel_neighbours[closest_pixel_indices[i]]] - vertices_z[i])**2, axis=1))) for i in range(len(vertices_z))]
            grad_z = np.array(distances_z) - np.array(distances)

            grad = [(grad_x[i], grad_y[i], grad_z[i]) for i in range(len(vertices))]    
            
            #### Compute normals and mean curvatures at every mesh vertex
            normals_filter = vtk.vtkPolyDataNormals()
            normals_filter.SetInputData(icosphere)
            normals_filter.ComputePointNormalsOn()
            normals_filter.ComputeCellNormalsOff()
            normals_filter.Update()
            normals = [normals_filter.GetOutput().GetPointData().GetNormals().GetTuple(i) for i in range(len(vertices))]
            
            # Curvatures
            curvatures_filter = vtk.vtkCurvatures()
            curvatures_filter.SetInputData(icosphere)
            curvatures_filter.SetCurvatureTypeToMean()
            curvatures_filter.Update()
            mean_curvatures = [curvatures_filter.GetOutput().GetPointData().GetScalars().GetTuple1(i) for i in range(len(vertices))]
            
            #### Compute scheme velocities based on distance gradient, normals and curvatures
            velocities = np.array([(np.dot(np.array(grad[i]), np.array(normals[i])) 
                        + 0.5 * mean_curvatures[i]*distances[i]
                        ) * np.array(normals[i]) 
                        * (distances[i]/np.max(distances)) 
                        for i in range(len(vertices))
                    ])
            #### Update vertex positions
            vertices = [np.array(vertices[i]) - neta[epoch] * np.array(velocities[i]) for i in range(len(vertices))]
            
            #### Update mesh
            for i in range(points.GetNumberOfPoints()):
                points.SetPoint(i, vertices[i])
            icosphere.SetPoints(points)
            icosphere.Modified()

            #end_time_epoch = time.time()
            #print("Cell " + str(cl) + "/" + str(nb_regions-1) + ", epoch " + str(epoch) + "/" + str(iterations) + f" processed in {end_time_epoch - start_time_epoch:.6f} seconds")
        end_time_cell = time.time()
        print(f"Cell {cl-1} / {len(regions_range)} processed in {end_time_cell - start_time_cell:.6f} seconds")

        cell_meshes+=[icosphere]

    return cell_meshes