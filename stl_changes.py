import os
import math
import numpy as np
from stl import mesh
import open3d as o3d
import copy


class stl_changes:

    def draw_registration_result(source, target, transformation):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp])


    def compute_transformation_matrix(stl_points, real_points):
        # 1. For each point in the source point cloud, match the closest point in the reference 
        # point cloud (by doing an euclidean distance for example).
        source = stl_points
        target = real_points


        # Run ICP registration
        threshold = 0.02  # Set a threshold for the point matching
        trans_init = np.identity(4)  # Start with an identity matrix as the initial transformation

        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())

        return reg_p2p

    def apply_transformation(stl_points, transformation_matrix):

        return transformed_points[:, :3]  # Convert back to 3D

    def track_stl_changes(source_stl, target_stl):
        # Load an STL mesh
        source_mesh = o3d.io.read_triangle_mesh("stl_files/cube.stl")
        target_mesh = o3d.io.read_triangle_mesh("stl_files/modified_cube.stl")

        # Convert the mesh to a point cloud
        source_point_cloud = source_mesh.sample_points_poisson_disk(number_of_points=1000)
        target_point_cloud = target_mesh.sample_points_poisson_disk(number_of_points=1000) 

        # Run ICP registration
        threshold = 0.02  # Set a threshold for the point matching
        trans_init = np.identity(4)  # Start with an identity matrix as the initial transformation
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source_point_cloud, target_point_cloud, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())

        draw_registration_result()

