from scipy.spatial.transform import Rotation as R
import pandas as pd
import numpy as np
import cv2
import time
import open3d as o3d
import os
from p1 import *
import glob 

def get_transform_mat_from_euler(rotation, translation, scale):
    #rotation:matrix
    r_mat = R.from_euler('xyz', rotation, degrees=True).as_matrix()
    scale_mat = np.eye(3) * scale
    transform_mat = np.concatenate([scale_mat @ r_mat, translation.reshape(3, 1)], axis=1)
    transform_mat = np.concatenate([transform_mat, np.asarray([[0,0,0,1]])], axis = 0)
    return transform_mat


def load_image(fname, images_df):
    image_path = "data/frames"
    idx = ((images_df.loc[images_df["NAME"] == fname])["IMAGE_ID"].values)[0]
    rimg = cv2.imread(os.path.join(image_path,fname))
    return idx, rimg

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
    gt_R = np.vstack((images_df["QW"].values, images_df["QX"].values, images_df["QY"].values, images_df["QZ"].values)).transpose((1,0))#(N,4)

    #create cube
    N = 8
    c_list = {1:(255, 0, 0), #red
              2:(0, 0, 255),   #blue
              3:(255, 168, 0), #orange
              4:(0, 255, 0),   #green
              5:(255, 255, 0), #yellow
              6:(100, 0, 255)}  #purple
    p_w = np.zeros((6, N, N, 3))
    points_color = np.zeros((6, N, N,1))
    length = 0.5
    cube = o3d.geometry.TriangleMesh.create_box(width=length, height=length, depth=length)
    cube_vertices = np.asarray(cube.vertices).copy()

    #for xplane
    for x in range(2):
        for i in range(N):
            for j in range(N):
                p_w[x, i , j,:] = np.asarray([[x*length, i*length/N + length/N*0.5, j*length/N + length/N*0.5]])
    #for y plane
    for y in range(2):
        for i in range(N):
            for j in range(N):
                p_w[2+y, i , j,:] = np.asarray([[i*length/N + length/N*0.5,y*length, j*length/N + length/N*0.5]])
    #for z plane

    for z in range(2):
        for i in range(N):
            for j in range(N):
                p_w[4+z, i , j,:] = np.asarray([[i*length/N + length/N*0.5, j*length/N + length/N*0.5, z*length]])
    for i in range(6):
        points_color[i,:, :] = i+1
    points_color = points_color.reshape((6*N*N,1)) 
    p_w = p_w.reshape((6*N*N,3))
    p_w[:,0]+=0.5
    p_w[:,1] += 0
    p_w[:, 2]-= 0.5

    
    p_w = np.concatenate([p_w, np.ones((6*N*N,1))], axis=1)
    cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])
    distCoeffs = np.array([0.0847023,-0.192929,-0.000201144,-0.000725352])

    filenames = glob.glob('data/frames/*jpg')
    filenames_sorted = []
    img_id_sorted = []
    img_idxs = []
    for fn in filenames:
        img_idxs.append(int(fn.split('/')[-1].split('.')[0].split('img')[-1]))
    img_idxs = np.asarray(img_idxs)
    img_idxs_argsort = np.argsort(img_idxs)
    for i in img_idxs_argsort:
        filenames_sorted.append(filenames[i])
        img_id_sorted.append(img_idxs[i])

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1080,1920))
    
    for i, fn in enumerate(filenames_sorted):
        fname = fn.split('/')[-1]
        idx, input_img = load_image(fname, images_df)

        #pw -> pc
        tvec = gt_t[idx-1].reshape(3,1) #(3,1)

        rot = R.from_quat(gt_R[idx-1]).as_matrix()

        H = np.concatenate([np.concatenate([rot, tvec], axis=1), np.asarray([[0,0,0,1]])], axis=0) #(4, 4)
        p_c = np.dot(H,p_w.transpose((1,0))).transpose((1,0))

        depth = p_c[:,2] #(N,)

        draw_order = np.argsort(-1*depth) #from far to near
        p_2d = np.dot(cameraMatrix, np.dot(np.concatenate([rot, tvec], axis=1), p_w.transpose((1,0)))).transpose((1,0)) #(N,3)
        for od in draw_order:
            u , v = (p_2d[od,:2]/p_2d[od, 2]).astype(int)

            if u>=0 and v>=0 and u<=1920 and v<=1080:
                cv2.circle(input_img,(u,v), 5, c_list[int(points_color[od,-1].item(0))], -1)

        
        #print(idx, fname,str(img_id_sorted[i]))
        out.write(input_img)
        #cv2.imwrite("p2/output-{}.jpg".format(str(img_id_sorted[i])), input_img)
    out.release()





   


    





