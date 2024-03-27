import os
import math
import numpy as np
from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt

new_working_directory = '/Users/numaanformoli/Documents/simulation_center/vr_project/Training-VR'
os.chdir(new_working_directory)

########## Plotting using matplotlib is equally easy #########
# figure = plt.figure()
# axes = figure.add_subplot(111, projection='3d')

# # Load the STL files and add the vectors to the plot
# your_mesh = mesh.Mesh.from_file('stl_files/Shape-Box.stl')
# axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors))

# # Auto scale to the mesh size
# scale = your_mesh.points.flatten()
# axes.auto_scale_xyz(scale, scale, scale)

# # Show the plot to the screen
# plt.show()

######### Modifying Mesh objects #########

# Create 3 faces of a cube
data = np.zeros(6, dtype=mesh.Mesh.dtype)


# Top of the cube
data['vectors'][0] = np.array([[0, 1, 1],
                                  [1, 0, 1],
                                  [0, 0, 1]])
data['vectors'][1] = np.array([[1, 0, 1],
                                  [0, 1, 1],
                                  [1, 1, 1]])
# Front face
data['vectors'][2] = np.array([[1, 0, 0],
                                  [1, 0, 1],
                                  [1, 1, 0]])
data['vectors'][3] = np.array([[1, 1, 1],
                                  [1, 0, 1],
                                  [1, 1, 0]])
# Left face
data['vectors'][4] = np.array([[0, 0, 0],
                                  [1, 0, 0],
                                  [1, 0, 1]])
data['vectors'][5] = np.array([[0, 0, 0],
                                  [0, 0, 1],
                                  [1, 0, 1]])

data['vectors'] -= .5

meshes = [mesh.Mesh(data.copy()) for _ in range(4)]
meshes[0].rotate([0.0, 0.5, 0.0], math.radians(90))


figure = plt.figure()
axes = figure.add_subplot(111, projection='3d')

your_mesh = mesh.Mesh(data) 
axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors))

scale = your_mesh.points.flatten()
axes.auto_scale_xyz(scale, scale, scale)

axes.set_xlabel('X Label')
axes.set_ylabel('Y Label')
axes.set_zlabel('Z Label')

plt.show()








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