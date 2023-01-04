import open3d as o3d
import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys, os
import pandas as pd

def load_point_cloud(points3D_df):

    xyz = np.vstack(points3D_df['XYZ'])
    rgb = np.vstack(points3D_df['RGB'])/255

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    
    return pcd

def load_pyramid():
    #CCS
    pyramid = o3d.geometry.LineSet()
    pyramid.points = o3d.utility.Vector3dVector([[-0.5, -0.5, -1], [0.5, -0.5, -1], [-0.5, 0.5, -1], [0.5,0.5,-1],[0, 0, 0]])
    pyramid.lines  = o3d.utility.Vector2iVector([[0, 1], [0, 2],[1,3],[2,3], [0, 4],[1,4],[2,4], [3,4]])          # X, Y, Z
    pyramid.colors = o3d.utility.Vector3dVector([[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]]) # R, G, B
    
    return pyramid

def get_transform_mat_from_M(rotation, translation, scale):
    #rotation:matrix
    #r_mat = R.from_euler('xyz', rotation, degrees=True).as_matrix()
    r_mat = rotation
    scale_mat = np.eye(3) * scale
    transform_mat = np.concatenate([scale_mat @ r_mat, translation.reshape(3, 1)], axis=1)
    transform_mat = np.concatenate([transform_mat, np.asarray([[0,0,0,1]])], axis = 0)
    return transform_mat

def get_transform_mat(rotation, translation, scale):
    r_mat = R.from_euler('xyz', rotation, degrees=True).as_matrix()
    scale_mat = np.eye(3) * scale
    transform_mat = np.concatenate([scale_mat @ r_mat, translation.reshape(3, 1)], axis=1)
    return transform_mat


def update_pyramid(rot, tvec):
    pyramid = load_pyramid()
    p_points = np.asarray(pyramid.points).copy()/10
    transform_mat = get_transform_mat_from_M(rot, tvec, 1)
    transform_vertices =  np.dot(np.linalg.inv(transform_mat), np.concatenate([p_points, np.ones((5,1))],axis=1).transpose()).transpose()
    pyramid.points = o3d.utility.Vector3dVector(transform_vertices[:, :3])

    return pyramid




if __name__ == '__main__':
    R_result = np.load("R_result.npy")
    print(R_result.shape)
    t_result = np.load("t_result.npy")
    print(t_result.shape)
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    # load point cloud
    points3D_df = pd.read_pickle("data/points3D.pkl")
    pcd = load_point_cloud(points3D_df)
    vis.add_geometry(pcd)
    

    
    for i, rot in enumerate(R_result):
        
        new_pyramid = update_pyramid(rot, t_result[i])       

        vis.add_geometry(new_pyramid)
    



    
    '''
    R_euler = np.array([0, 0, 0]).astype(float)
    t = np.array([0, 0, 0]).astype(float)
    scale = 1.0
    update_cube()

    '''

    # just set a proper initial camera view
    vc = vis.get_view_control()
    vc_cam = vc.convert_to_pinhole_camera_parameters()
    initial_cam = get_transform_mat(np.array([7.227, -16.950, -14.868]), np.array([-0.351, 1.036, 5.132]), 1)
    initial_cam = np.concatenate([initial_cam, np.zeros([1, 4])], 0)
    initial_cam[-1, -1] = 1.
    setattr(vc_cam, 'extrinsic', initial_cam)
    vc.convert_from_pinhole_camera_parameters(vc_cam)



    vis.run()
    vis.destroy_window()

    '''
    print('Rotation matrix:\n{}'.format(R.from_euler('xyz', R_euler, degrees=True).as_matrix()))
    print('Translation vector:\n{}'.format(t))
    print('Scale factor: {}'.format(scale))
    '''

    #np.save('cube_transform_mat.npy', get_transform_mat(R_euler, t, scale))
    #np.save('cube_vertices.npy', np.asarray(cube.vertices))
    