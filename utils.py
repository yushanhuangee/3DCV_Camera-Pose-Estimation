import numpy as np
import cv2
from sklearn.linear_model import LinearRegression




def RANSAC(points2D, points3D, top_k=0):
    sample = 10
    N_step = 20
    thre = 3
    p = 0.99
    point_num = points2D.shape[0]
    selected_points2d = points2D
    selected_points3d = points3D


    best_ratio = 0

    for i in range(N_step):
        #random select
        idx = np.random.choice(point_num, sample)
        x = points2D[idx,:]
        y = points3D[idx,:]

        #fitting
        reg = LinearRegression()
        reg.fit(x, y)
        point3d_pred = reg.predict(points2D)

        #compute inlier
        error = np.sum((points3D-point3d_pred)**2, axis=1)/points3D.shape[1]
        inlier = error < thre
        ratio = np.count_nonzero(inlier)/point_num
        
        inlier_idx = np.squeeze(np.asarray(np.nonzero(inlier)))

        #check probability or if inlier contain 1/2 points
        if ratio >= p:            
            selected_points2d = points2D[inlier_idx]            
            selected_points3d = points3D[inlier_idx]
            #matching_coef = reg.coef_
            break

        if ratio > best_ratio:
            best_ratio = ratio
            selected_points2d = points2D[inlier_idx]            
            selected_points3d = points3D[inlier_idx]
            #matching_coef = reg.coef_

    return selected_points2d[:,:], selected_points3d[:,:]

def get_homo_from_x(x):
    '''
    get homo 2d from xy
    Input:
    - xy		type: torch.Tensor (N, 2)
    Return:
    - homo		type: torch.Tensor (N, 3)
    get homo 3d from xyz
    Input:
    - xyz		type: torch.Tensor (N, 3)
    Return:
    - homo		type: torch.Tensor (N, 4)
    '''
    N = x.shape[0]
    homo_ones = np.ones((N, 1))
    homo_2d = np.concatenate((x, homo_ones), axis=1)
    return homo_2d
