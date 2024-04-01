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
    
    def create_triangle(self):
        # Define the vertices
        vertices = np.array([\
            [-1, -1, -1],
            [+1, -1, -1],
            [+1, +1, -1]])

        triangle = mesh.Mesh(np.zeros(1, dtype=mesh.Mesh.dtype))
        for i, v in enumerate(vertices):
            triangle.vectors[0][i] = v

        return triangle
    
    # Affine transformation
    # Input: expects 4x4 matrix of points
    # Returns A,t
    # A = 3x3 rotation matrix
    # t = 3x1 column vector
    def transformation_matrix_method_1(self,source_coord_points, target_coord_points):

        # Calculate the transformation
        l = len(source_coord_points)
        print("source_coord_points: ", source_coord_points)
        B = np.vstack([np.transpose(source_coord_points), np.ones(l)])
        print(B)
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

    # Rigid transformation
    # Input: expects 3xN matrix of points
    # Returns R,t
    # R = 3x3 rotation matrix
    # t = 3x1 column vector
    def transformation_matrix_method_2(self, A, B):

        # Transpose points to match function's expected input format
        A = np.array(A).T
        B = np.array(B).T

        assert A.shape == B.shape

        num_rows, num_cols = A.shape
        if num_rows != 3:
            raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

        num_rows, num_cols = B.shape
        if num_rows != 3:
            raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

        # find mean column wise
        centroid_A = np.mean(A, axis=1)
        centroid_B = np.mean(B, axis=1)

        # ensure centroids are 3x1
        centroid_A = centroid_A.reshape(-1, 1)
        centroid_B = centroid_B.reshape(-1, 1)

        # subtract mean
        Am = A - centroid_A
        Bm = B - centroid_B

        H = Am @ np.transpose(Bm)

        # sanity check
        #if linalg.matrix_rank(H) < 3:
        #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

        # find rotation
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # special reflection case
        if np.linalg.det(R) < 0:
            print("det(R) < R, reflection detected!, correcting for it ...")
            Vt[2,:] *= -1
            R = Vt.T @ U.T

        t = -R @ centroid_A + centroid_B

        return R, t

    # Using 4 points and a transformation matrix
    def stl_changes_four_point_cube(self, source_coord_points, target_coord_points):
        original_cube = self.create_cube()
        modified_cube = self.create_cube()

        figure = plt.figure()
        axes = figure.add_subplot(111, projection='3d')

        # do a bunch of transformations
        modified_cube.rotate([0.5, 0.0, 0.0], math.radians(30))
        modified_cube.x += 2
        modified_cube.y += 2
        print(modified_cube.vectors)

        # Original Positions Graphed
        axes.add_collection3d(mplot3d.art3d.Poly3DCollection(modified_cube.vectors))
        axes.add_collection3d(mplot3d.art3d.Poly3DCollection(original_cube.vectors))

        scale = np.concatenate(original_cube.points + modified_cube.points).flatten()
        axes.auto_scale_xyz(scale, scale, scale)

        plt.show()

        # Apply transformation
        rotation_matrix, translation = self.transformation_matrix_method_2(source_coord_points, target_coord_points)
        original_cube.rotate_using_matrix(rotation_matrix)
        original_cube.translate(translation)

        # New Positions Graphed
        figure = plt.figure()
        axes = figure.add_subplot(111, projection='3d')

        axes.add_collection3d(mplot3d.art3d.Poly3DCollection(modified_cube.vectors))
        axes.add_collection3d(mplot3d.art3d.Poly3DCollection(original_cube.vectors))

        scale = np.concatenate(original_cube.points + modified_cube.points).flatten()
        axes.auto_scale_xyz(scale, scale, scale)

        plt.show()

    def stl_changes_four_point_triangle(self, source_coord_points, target_coord_points):
        original_cube = self.create_triangle()
        modified_cube = self.create_triangle()

        figure = plt.figure()
        axes = figure.add_subplot(111, projection='3d')

        # do a bunch of transformations
        modified_cube.rotate([0.5, 0.0, 0.0], math.radians(90))
        modified_cube.x += 2
        modified_cube.y += 2
        print(modified_cube.vectors)

        # Original Positions Graphed
        axes.add_collection3d(mplot3d.art3d.Poly3DCollection(modified_cube.vectors))
        axes.add_collection3d(mplot3d.art3d.Poly3DCollection(original_cube.vectors))

        scale = np.concatenate(original_cube.points + modified_cube.points).flatten()
        axes.auto_scale_xyz(scale, scale, scale)

        plt.show()

        # Apply transformation
        rotation_matrix, translation = self.transformation_matrix_method_2(source_coord_points, target_coord_points)
        original_cube.rotate_using_matrix(rotation_matrix)
        original_cube.translate(translation)

        # New Positions Graphed
        figure = plt.figure()
        axes = figure.add_subplot(111, projection='3d')

        axes.add_collection3d(mplot3d.art3d.Poly3DCollection(modified_cube.vectors))
        axes.add_collection3d(mplot3d.art3d.Poly3DCollection(original_cube.vectors))

        scale = np.concatenate(original_cube.points + modified_cube.points).flatten()
        axes.auto_scale_xyz(scale, scale, scale)

        plt.show()



new_working_directory = '/Users/numaanformoli/Documents/simulation_center/vr_project/Training-VR'
os.chdir(new_working_directory)
# stl_tracker = stl_changes()
# stl_tracker.track_stl_changes_point_cloud("stl_files/cube.stl", "stl_files/modified_cube.stl")

stl_tracker = stl_changes()

# Tracking of 3D Object
# stl_tracker.stl_changes_four_point([[-1, -1, -1], [+1, -1, -1], [+1, -1, +1], [+1, +1, +1]], [[1, 0.7182872, 0.59767246], [3, 0.7182872, 0.59767246], [3, 2.5976725, 1.2817128], [3, 3.2817128, -0.59767246]])
# stl_tracker.stl_changes_four_point([[-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, -1, +1]], [[1, 1, 1], [1, 1, -1], [3, 1, 1], [1, 1, -1]])

# Tracking of 2D Object
stl_tracker.stl_changes_four_point_triangle([[-1, -1, -1], [+1, -1, -1], [+1, +1, -1]], [[ 1,  1,  1], [ 3,  1,  1], [ 3,  1, -1]])
