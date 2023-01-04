from scipy.spatial.transform import Rotation as R
import pandas as pd
import numpy as np
import cv2
import time
from utils import *
from Epnp import *
import open3d as o3d

def average(x):
    return list(np.mean(x,axis=0))

def average_desc(train_df, points3D_df):
    train_df = train_df[["POINT_ID","XYZ","RGB","DESCRIPTORS"]]
    desc = train_df.groupby("POINT_ID")["DESCRIPTORS"].apply(np.vstack)
    desc = desc.apply(average)
    desc = desc.reset_index()
    desc = desc.join(points3D_df.set_index("POINT_ID"), on="POINT_ID")
    return desc




def solveEPnP(points3D, points2D, cameraMatrix, distCoeffs):
    points2D, points3D = RANSAC(points2D, points3D)
    epnp = EPnP(cameraMatrix, points2D, points3D)
    R, t = epnp.compute_Pose()    
    return R, t

def pnpsolver(query,model,cameraMatrix=0,distortion=0):
    kp_query, desc_query = query
    kp_model, desc_model = model

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc_query,desc_model,k=2)

    gmatches = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            gmatches.append(m)

    points2D = np.empty((0,2))
    points3D = np.empty((0,3))

    for mat in gmatches:
        query_idx = mat.queryIdx
        model_idx = mat.trainIdx
        points2D = np.vstack((points2D, kp_query[query_idx]))
        points3D = np.vstack((points3D, kp_model[model_idx]))

    cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])    
    distCoeffs = np.array([0.0847023,-0.192929,-0.000201144,-0.000725352])
    return solveEPnP(points3D, points2D, cameraMatrix, distCoeffs)
    #return cv2.solvePnPRansac(points3D, points2D, cameraMatrix, distCoeffs)

def computeRterror(rotq, tvec, gt_R, gt_t, R_type = "rotM" ):
    #print(gt_t)

    tvec = tvec.reshape((3,))
    #print(tvec)

    #print(rotq)
    #rotation error
        
    if R_type == "rotM":
        rotq_inv = R.from_matrix(rotq).inv()
    else:
        rotq_inv = R.from_quat(rotq.squeeze()).inv()
    gt_R_R = R.from_quat(gt_R)
    dr_matrix = np.matmul(gt_R_R.as_matrix(), rotq_inv.as_matrix())
    
    #print(gt_R_R.as_matrix())
    dr_R = R.from_matrix(dr_matrix)
    rotation_error = np.linalg.norm(dr_R.as_rotvec())

    print(rotation_error)

    #translation error
    translation_error = np.linalg.norm((gt_t-tvec))
    #print(gt_t-tvec)
    print(translation_error)
    return rotation_error, translation_error
    

def load_query(idx, images_df, point_desc_df):
    fname = ((images_df.loc[images_df["IMAGE_ID"] == idx])["NAME"].values)[0]
    rimg = cv2.imread("data/frames/"+fname,cv2.IMREAD_GRAYSCALE)

    points = point_desc_df.loc[point_desc_df["IMAGE_ID"]==idx]
    kp_query = np.array(points["XY"].to_list())
    desc_query = np.array(points["DESCRIPTORS"].to_list()).astype(np.float32)
    return rimg, kp_query, desc_query

if __name__ == '__main__':
    images_df = pd.read_pickle("data/images.pkl")
    train_df = pd.read_pickle("data/train.pkl")
    points3D_df = pd.read_pickle("data/points3D.pkl")
    point_desc_df = pd.read_pickle("data/point_desc.pkl")


    # Process model descriptors
    desc_df = average_desc(train_df, points3D_df)
    kp_model = np.array(desc_df["XYZ"].to_list())
    desc_model = np.array(desc_df["DESCRIPTORS"].to_list()).astype(np.float32)
    idxs = images_df["IMAGE_ID"].values


    #rotation ground-truth
    gt_t = np.vstack((images_df["TX"].values, images_df["TY"].values, images_df["TZ"].values)).transpose((1,0)) #(N,3)
    gt_R = np.vstack((images_df["QX"].values, images_df["QY"].values, images_df["QZ"].values, images_df["QW"].values)).transpose((1,0))#(N,4)

    R_errors = []
    t_errors = []
    rotq_record = []
    tvec_record = []
    gt_rotq_record =[]
    gt_tvec_record=[]
    t1 = time.time()

    # Load query image
    for idx in idxs:
        print(idx)
        rimg, kp_query, desc_query = load_query(idx, images_df, point_desc_df)

        # Find correspondance and solve pnp
        
        
        '''
        retval, rvec, tvec, inliers = pnpsolver((kp_query, desc_query),(kp_model, desc_model))
        #rotq = R.from_rotvec(rvec.reshape(1,3)).as_quat()

        rotq = R.from_rotvec(rvec.reshape(3,)).as_matrix()

        tvec = tvec.reshape(1,3).transpose((1,0))
        '''
        
        rotq, tvec = pnpsolver((kp_query, desc_query),(kp_model, desc_model))
        rotation_error, translation_error = computeRterror(rotq, tvec, gt_R[idx-1], gt_t[idx-1])
        R_errors.append(rotation_error)
        t_errors.append(translation_error)
        rotq_record.append(rotq)
        tvec_record.append(tvec)
        '''
        tmp_R = R.from_quat(gt_R[idx-1]).as_matrix()
        gt_rotq_record.append(tmp_R)
        gt_tvec_record.append(gt_t[idx-1].reshape(3,1))
        '''



    
    t2 = time.time()
    print('time elapsed: ' + str(round(t2-t1, 2)) + ' seconds')
    print('time elapsed: ' + str(t2-t1) + ' seconds')

    np.save('R_result', rotq_record)
    np.save('t_result', tvec_record)
    #computemedian
    print("R_errors_median:" ,np.median(np.asarray(R_errors)))
    print("t_errors_median:", np.median(np.asarray(t_errors)))



    

    
