import os
import time
import copy
import torch
import torchvision
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms, models, datasets
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from dataset import CustomDataset
from torchsummary import summary
from models import PoseNet
import time
from scipy.spatial.transform import Rotation as R

def computeRterror(rotq, tvec, gt_R, gt_t):
    #print(gt_t)

    tvec = tvec.reshape((3,))
    #print(tvec)

    #print(rotq)
    #rotation error
    rotq_inv = R.from_matrix(rotq).inv()

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

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize(300),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = CustomDataset('train', transform)
    val_dataset = CustomDataset("valid", transform)

    print(device)

    train_data_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_data_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    base_model = models.inception_v3(pretrained=True)
    base_model.aux_logits = False
    model = PoseNet(base_model)

    model.load_state_dict(torch.load("29_net.pth"))
    

    model = model.to(device)
    '''
    inputs, poses = next(iter(train_data_loader))
    out = torchvision.utils.make_grid(inputs)
    imshow(out, 'sample image')
    '''

    

    # Setup for tensorboard
    writer = SummaryWriter()

    
    n_iter = 0
    model.eval()
    
    rotq_record = []
    tvec_record = []

    R_errors = []
    t_errors = []
    t1 = time.time()


    for i, (inputs, gt_t, gt_R) in enumerate(train_data_loader):
        inputs = inputs.to(device)

        # forward
        pos_out, ori_out = model(inputs)

        pos_true = gt_t
        ori_true = gt_R

        beta = 500
        #ori_out = F.normalize(ori_out, p=2, dim=1)
        #ori_true = F.normalize(ori_true, p=2, dim=1)


        #transfrom to rotation matrix

        rot = R.from_quat(ori_out.cpu().detach().numpy()).as_matrix()
        tvec = pos_out.cpu().detach().numpy()


        rotq_record.append(rot)
        tvec_record.append(tvec)

        
        rotation_error, translation_error = computeRterror(rot, tvec, ori_true.cpu().detach().numpy(), pos_true.cpu().detach().numpy())
        R_errors.append(rotation_error)
        t_errors.append(translation_error)      

        
    for i, (inputs, gt_t, gt_R) in enumerate(val_data_loader):

        inputs = inputs.to(device)

        # forward
        pos_out, ori_out = model(inputs)

        pos_true = gt_t
        ori_true = gt_R

        beta = 500
        ori_out = F.normalize(ori_out, p=2, dim=1)
        ori_true = F.normalize(ori_true, p=2, dim=1)


        #transfrom to rotation matrix

        rot = R.from_quat(ori_out.cpu().detach().numpy()).as_matrix()
        tvec = pos_out.cpu().detach().numpy()


        rotq_record.append(rot)
        tvec_record.append(tvec)

        
        rotation_error, translation_error = computeRterror(rot, tvec, ori_true.cpu().detach().numpy(), pos_true.cpu().detach().numpy())
        R_errors.append(rotation_error)
        t_errors.append(translation_error) 

    t2 = time.time()
    print('time elapsed: ' + str(round(t2-t1, 2)) + ' seconds')
    print('time elapsed: ' + str(t2-t1) + ' seconds')

    print(np.asarray(rotq_record).squeeze().shape)

    np.save('R_result_posenet', np.asarray(rotq_record).squeeze())
    np.save('t_result_posenet', np.asarray(tvec_record).squeeze())
    #computemedian
    print("R_errors_median:" ,np.median(np.asarray(R_errors)))
    print("t_errors_median:", np.median(np.asarray(t_errors)))
    print(np.asarray(R_errors).shape)

    print(np.asarray(t_errors).shape)

