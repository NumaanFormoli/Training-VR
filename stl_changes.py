import copy
import os
import math
import numpy as np
from stl import mesh
import open3d as o3d
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt


class stl_changes:

    def draw_registration_result(self, source_mesh, target_mesh, transformation):
        source_temp = copy.deepcopy(source_mesh)
        target_temp = copy.deepcopy(target_mesh)
        o3d.visualization.draw_geometries([source_temp, target_temp])
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp])


    # Using point set registration
    def track_stl_changes_point_cloud(self, source_stl, target_stl):
        # Load an STL mesh
        source_mesh = o3d.io.read_triangle_mesh("stl_files/cube.stl")
        target_mesh = o3d.io.read_triangle_mesh("stl_files/modified_cube.stl")

        # Convert the mesh to a point cloud
        source_point_cloud = source_mesh.sample_points_poisson_disk(number_of_points=1000)
        # source_point_cloud = source_point_cloud.voxel_down_sample(70)


        target_point_cloud = target_mesh.sample_points_poisson_disk(number_of_points=1000)

        # translation_vec = source_point_cloud.get_center() - target_point_cloud.get_center()
        # target_point_cloud.translate(translation_vec)

        # Run ICP registration
        threshold = 25  # Set a threshold for the point matching
        trans_init = np.identity(4)  # Start with an identity matrix as the initial transformation
        print("Initial Alignment:")
        evaluation = o3d.pipelines.registration.evaluate_registration(source_point_cloud, target_point_cloud, threshold, trans_init)
        print(evaluation)
        reg_p2p = o3d.pipelines.registration.registration_icp(source_point_cloud, target_point_cloud, threshold, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPoint())
        print("Final Alignment")
        print(reg_p2p)
        self.draw_registration_result(source_mesh, target_mesh, reg_p2p.transformation)

    def create_cube(self):
        # Define the 8 vertices of the cube
        vertices = np.array([\
            [-1, -1, -1],
            [+1, -1, -1],
            [+1, +1, -1],
            [-1, +1, -1],
            [-1, -1, +1],
            [+1, -1, +1],
            [+1, +1, +1],
            [-1, +1, +1]])
        # Define the 12 triangles composing the cube
        faces = np.array([\
            [0,3,1],
            [1,3,2],
            [0,4,7],
            [0,7,3],
            [4,5,6],
            [4,6,7],
            [5,1,2],
            [5,2,6],
            [2,3,6],
            [3,7,6],
            [0,1,5],
            [0,5,4]])

        # Create the mesh
        cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                cube.vectors[i][j] = vertices[f[j],:]

        # Write the mesh to file "cube.stl"
        return cube

    def transformation_matrix(self,source_coord_points, target_coord_points):

        # Calculate the transformation
        l = len(source_coord_points)
        B = np.vstack([np.transpose(source_coord_points), np.ones(l)])
        D = 1.0 / np.linalg.det(B)
        entry = lambda r,d: np.linalg.det(np.delete(np.vstack([r, B]), (d+1), axis=0))
        M = [[(-1)**i * D * entry(R, i) for i in range(l)] for R in np.transpose(target_coord_points)]
        A, t = np.hsplit(np.array(M), [l-1])
        t = np.transpose(t)[0]

        # output
        print("Affine transformation matrix:\n", A)
        print("Affine transformation translation vector:\n", t)
        print("TESTING:")
        for p, P in zip(np.array(source_coord_points), np.array(target_coord_points)):
            image_p = np.dot(A, p) + t
            result = "[OK]" if np.allclose(image_p, P) else "[ERROR]"
            print(p, " mapped to: ", image_p, " ; expected: ", P, result)

        return A, t

    

    # Using 4 points and a transformation matrix
    def stl_changes_four_point(self, source_coord_points, target_coord_points):
        original_cube = self.create_cube()
        modified_cube = self.create_cube()

        figure = plt.figure()
        axes = figure.add_subplot(111, projection='3d')

        axes.add_collection3d(mplot3d.art3d.Poly3DCollection(modified_cube.vectors))
        axes.add_collection3d(mplot3d.art3d.Poly3DCollection(original_cube.vectors))

        # Auto scale to the mesh size
        scale = np.concatenate(original_cube.points + modified_cube.points).flatten()
        axes.auto_scale_xyz(scale, scale, scale)

        # Show the plot to the screen
        plt.show()


        # do a bunch of transformations
        modified_cube.rotate([0.5, 0.0, 0.0], math.radians(90))
        modified_cube.x += 2
        modified_cube.y += 2
                
        # Apply transformation
        rotation_matrix, translation = self.transformation_matrix(source_coord_points, target_coord_points)
        original_cube.rotate_using_matrix(rotation_matrix)
        original_cube.translate(translation)

        # Graph both cubes
        figure = plt.figure()
        axes = figure.add_subplot(111, projection='3d')

        axes.add_collection3d(mplot3d.art3d.Poly3DCollection(modified_cube.vectors))
        axes.add_collection3d(mplot3d.art3d.Poly3DCollection(original_cube.vectors))

        # Auto scale to the mesh size
        scale = np.concatenate(original_cube.points + modified_cube.points).flatten()
        axes.auto_scale_xyz(scale, scale, scale)

        # Show the plot to the screen
        plt.show()


new_working_directory = '/Users/numaanformoli/Documents/simulation_center/vr_project/Training-VR'
os.chdir(new_working_directory)
# stl_tracker = stl_changes()
# stl_tracker.track_stl_changes_point_cloud("stl_files/cube.stl", "stl_files/modified_cube.stl")

stl_tracker = stl_changes()
stl_tracker.stl_changes_four_point([[-1, -1, -1], [+1, -1, -1], [+1, -1, +1], [+1, +1, +1]], [[1, 0.7182872, 0.59767246], [3, 0.7182872, 0.59767246], [3, 2.5976725, 1.2817128], [3, 3.2817128, -0.59767246]])
