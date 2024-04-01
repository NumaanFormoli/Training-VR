import os
import math
import numpy as np
from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
import open3d as o3d

new_working_directory = '/Users/numaanformoli/Documents/simulation_center/vr_project/Training-VR'
os.chdir(new_working_directory)

# # Load an STL mesh
# mesh = o3d.io.read_triangle_mesh("stl_files/cube.stl")

# # Convert the mesh to a point cloud
# point_cloud = mesh.sample_points_poisson_disk(number_of_points=1000)

# # extract xyz coordinates
# points = np.asarray(point_cloud.points)

# Visualize the point cloud
# o3d.visualization.draw_geometries([point_cloud])

# mesh.compute_vertex_normals()

# # Step 3: Visualize the mesh
# o3d.visualization.draw_geometries([mesh])




########## Plotting using matplotlib is equally easy ##########
# figure = plt.figure()
# axes = figure.add_subplot(111, projection='3d')

# # Load the STL files and add the vectors to the plot
# your_mesh = mesh.Mesh.from_file('stl_files/cube.stl')
# axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors))

# # Auto scale to the mesh size
# scale = your_mesh.points.flatten()
# axes.auto_scale_xyz(scale, scale, scale)

# # Show the plot to the screen
# plt.show()

######### Modifying Mesh objects #########

# # Create 3 faces of a cube
# data = np.zeros(6, dtype=mesh.Mesh.dtype)

# # Top of the cube
# data['vectors'][0] = np.array([[0, 1, 1],
#                                   [1, 0, 1],
#                                   [0, 0, 1]])
# data['vectors'][1] = np.array([[1, 0, 1],
#                                   [0, 1, 1],
#                                   [1, 1, 1]])
# # Front face
# data['vectors'][2] = np.array([[1, 0, 0],
#                                   [1, 0, 1],
#                                   [1, 1, 0]])
# data['vectors'][3] = np.array([[1, 1, 1],
#                                   [1, 0, 1],
#                                   [1, 1, 0]])
# # Left face
# data['vectors'][4] = np.array([[0, 0, 0],
#                                   [1, 0, 0],
#                                   [1, 0, 1]])
# data['vectors'][5] = np.array([[0, 0, 0],
#                                   [0, 0, 1],
#                                   [1, 0, 1]])

# data['vectors'] -= 0.5

# # Generate 4 different meshes so we can rotate them later
# meshes = [mesh.Mesh(data.copy()) for _ in range(4)]

# # Rotate 90 degrees over the Y axis
# meshes[0].rotate([0.0, 0.5, 0.0], math.radians(90))

# # Translate 2 points over the X axis
# meshes[1].x += 2

# # Rotate 90 degrees over the X axis
# meshes[2].rotate([0.5, 0.0, 0.0], math.radians(90))

# # Translate 2 points over the X and Y points
# meshes[2].x += 2
# meshes[2].y += 2

# # Rotate 90 degrees over the X and Y axis
# meshes[3].rotate([0.5, 0.0, 0.0], math.radians(90))
# meshes[3].rotate([0.0, 0.5, 0.0], math.radians(90))
# # Translate 2 points over the Y axis
# meshes[3].y += 2


# figure = plt.figure()
# axes = figure.add_subplot(111, projection='3d')

# for m in meshes:
#     axes.add_collection3d(mplot3d.art3d.Poly3DCollection(m.vectors))
#     print(m.points)

# # Want to include them in the scale because the plot does not pick them up otherwise
# scale = np.concatenate([m.points for m in meshes]).flatten()
# axes.auto_scale_xyz(scale, scale, scale)

# axes.set_xlabel('X Label')
# axes.set_ylabel('Y Label')
# axes.set_zlabel('Z Label')

# plt.show()


######### Extending Mesh objects #########

# # Create 3 faces of a cube
# data = np.zeros(6, dtype=mesh.Mesh.dtype)

# # Top of the cube
# data['vectors'][0] = np.array([[0, 1, 1],
#                                   [1, 0, 1],
#                                   [0, 0, 1]])
# data['vectors'][1] = np.array([[1, 0, 1],
#                                   [0, 1, 1],
#                                   [1, 1, 1]])
# # Front face
# data['vectors'][2] = np.array([[1, 0, 0],
#                                   [1, 0, 1],
#                                   [1, 1, 0]])
# data['vectors'][3] = np.array([[1, 1, 1],
#                                   [1, 0, 1],
#                                   [1, 1, 0]])
# # Left face
# data['vectors'][4] = np.array([[0, 0, 0],
#                                   [1, 0, 0],
#                                   [1, 0, 1]])
# data['vectors'][5] = np.array([[0, 0, 0],
#                                   [0, 0, 1],
#                                   [1, 0, 1]])


# cube_front = mesh.Mesh(data.copy())
# cube_back = mesh.Mesh(data.copy());

# cube_back.rotate([0.5, 0.0, 0.0], math.radians(90))
# cube_back.rotate([0.0, 0.5, 0.0], math.radians(90))
# cube_back.rotate([0.5, 0.0, 0.0], math.radians(90))


# cube = mesh.Mesh(np.concatenate([
#     cube_back.data.copy(),
#     cube_front.data.copy(),
# ]))

# print(cube_front.vectors)


######### Extending Mesh objects #########

# # Define the 8 vertices of the cube
# vertices = np.array([\
#     [-1, -1, -1],
#     [+1, -1, -1],
#     [+1, +1, -1],
#     [-1, +1, -1],
#     [-1, -1, +1],
#     [+1, -1, +1],
#     [+1, +1, +1],
#     [-1, +1, +1]])
# # Define the 12 triangles composing the cube
# faces = np.array([\
#     [0,3,1],
#     [1,3,2],
#     [0,4,7],
#     [0,7,3],
#     [4,5,6],
#     [4,6,7],
#     [5,1,2],
#     [5,2,6],
#     [2,3,6],
#     [3,7,6],
#     [0,1,5],
#     [0,5,4]])

# # Create the mesh
# cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
# for i, f in enumerate(faces):
#     for j in range(3):
#         cube.vectors[i][j] = vertices[f[j],:]

# # Write the mesh to file "cube.stl"
# cube.save('cube.stl')


# MESHGRID
# def fun(x, y):
#     return x**2 + y

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# x = y = np.arange(-3.0, 3.0, 0.05)
# X, Y = np.meshgrid(x, y)
# zs = np.array(fun(np.ravel(X), np.ravel(Y)))
# Z = zs.reshape(X.shape)

# ax.plot_surface(X, Y, Z)

# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')

# plt.show()