import time
import morphonet
import numpy as np
import vtkmodules.all as vtk
from scipy.spatial import KDTree
from skimage import morphology,measure
import src.meshutils as mutils


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

def generate_regular_meshes(image_file, subdivisions, iterations, neta, cell_surfaces, previous_meshes_space, previous_meshes_time):

    image = morphonet.tools.imread(image_file)
    #labeled_image = measure.label(image)
    regions = measure.regionprops(image)

    cell_meshes = {}
    nb_regions = len(regions)+1
    regions_range = range(2, nb_regions) if cell_surfaces else range(1,2)
    
    for r in regions:
        cl = r["label"]

        if cl > 1:
            # Iterate through the labeled regions (excluding background)
            start_time_cell = time.time()
        
            #### Comupte surface pixels
            cell_mask = image == cl
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
            icosphere = mutils.create_icosphere(max_radius,centre, subdivisions)

            # Initialise with the previous state of the cell // probably need to pass the previous computed meshes as parameters
            # Works well with embryo surfaces, for cellular surfaces, the lineage is needed. This has not been implemented yet in this case

            if not cell_surfaces:
                if len(previous_meshes_space) == 0 and len(previous_meshes_time) == 0:
                    icosphere = mutils.create_icosphere(max_radius, centre, subdivisions)
                if len(previous_meshes_space) == 0 and len(previous_meshes_time) > 0:
                    icosphere = mutils.create_icosphere2(max_radius, centre, subdivisions, previous_meshes_time[cl-1])
                if len(previous_meshes_space) > 0 and len(previous_meshes_time) == 0:
                    icosphere = mutils.subdivide_mesh(previous_meshes_space[cl-1])
                if len(previous_meshes_space) > 0 and len(previous_meshes_time) > 0:
                    icosphere = mutils.subdivide_mesh2(previous_meshes_space[cl-1], previous_meshes_time[cl-1])

            #if not cell_surfaces:
            #    if len(previous_meshes) > 0 and not subdivide_previous_meshes:
            #        icosphere = mutils.create_icosphere2(max_radius, centre, subdivisions, previous_meshes[cl-1])
            #    if len(previous_meshes) > 0 and subdivide_previous_meshes:
            #        icosphere = mutils.subdivide_mesh(previous_meshes[cl-1])
            
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

                #nabla = mutils.compute_grad_operator(icosphere)
                #grad1_x = np.dot(nabla[:,:,0], distances)
                #grad1_y = np.dot(nabla[:,:,1], distances)
                #grad1_z = np.dot(nabla[:,:,2], distances)
                #print(grad_x.shape)
                
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

                grad = [np.array([grad_x[i], grad_y[i], grad_z[i]]) for i in range(len(vertices))]
                #grad = [np.array([grad1_x[i], grad1_y[i], grad1_z[i]]) for i in range(len(vertices))]
                #grad = np.array([grad[i]/np.linalg.norm(grad[i]) for i in range(len(vertices))])
                #print(distances[1], grad[1], grad1[1])    
                
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
                            ) 
                            * np.array(normals[i]) 
                            * (distances[i]/np.max(distances)) 
                            for i in range(len(vertices))
                        ])
                
                velocities = mutils.smooth_field(icosphere, velocities)

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

            cell_meshes[cl]=icosphere

    return cell_meshes
