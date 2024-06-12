import os
import time

root_folder = "nii/" #Put all embryo folders in this directory
#root_folder = "../../homeosphere/DATA/"

folders = [
    #"Astec-Phmamm1-v2/HomeoSphere_OBJ/cells/",
    #"Astec-Phmamm3-v2/HomeoSphere_OBJ/cells/",
    #"Astec-Phmamm4-v2/HomeoSphere_OBJ/cells/",
    #"Astec-Phmamm5-v2/HomeoSphere_OBJ/cells/",
    #"Astec-Phmamm7-v2/HomeoSphere_OBJ/cells/",
    #"Astec-Phmamm8-v2/HomeoSphere_OBJ/cells/",
    #"Astec-Phmamm9-v2/HomeoSphere_OBJ/cells/",
    "embryo1",
    #"embryo2",
    #"Astec-Pm1"
]

#folders = os.listdir(root_folder)
#folders = [f for f in folders if os.path.isdir(os.path.join(root_folder, f)) and not f.startswith('.')]

start_time_batch = time.time()
#command = ""
for folder in folders:
    
    start_time_folder = time.time()

    folder_path = root_folder + folder + "/"
    
    cell_surfaces = False # If True, computes cellular surfaces, if False, computes embryonic surface
    subdivisions = 2
    iterations = 125
    
    if not cell_surfaces:
        subdivisions = 3
    
    #command += "python3 main.py " + (folder_path) + " " + str(subdivisions) + " " + str(iterations) + " " + str(cell_surfaces) + " &"
    os.system("python3 main.py " + (folder_path) + " " + str(subdivisions) + " " + str(iterations) + " " + str(cell_surfaces))
    #os.system("python3 karim.py " + (folder_path))

    end_time_folder = time.time()

    elapsed_time = end_time_folder - start_time_folder
    print(f"Execution time for {folder}: {elapsed_time:.6f} seconds")

end_time_batch = time.time()

print(f"Total execution time: {end_time_batch - start_time_batch:.6f} seconds")