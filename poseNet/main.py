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


val_frequency = 10

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

    train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    base_model = models.inception_v3(pretrained=True)
    base_model.aux_logits = False
    model = PoseNet(base_model)
    

    model = model.to(device)  
    model.load_state_dict(torch.load("models/46_net.pth"))
    '''
    inputs, poses = next(iter(train_data_loader))
    out = torchvision.utils.make_grid(inputs)
    imshow(out, 'sample image')
    '''

    optimizer = optim.Adam(model.parameters(), lr=0.000001)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=50)

    num_epochs = 80

    # Setup for tensorboard
    writer = SummaryWriter()

    since = time.time()
    n_iter = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-'*20)        
        model.train()

        for i, (inputs, gt_t, gt_R) in enumerate(train_data_loader):

            inputs = inputs.to(device)
            gt_t = gt_t.to(device)
            gt_R = gt_R.to(device)

            # Zero the parameter gradient
            optimizer.zero_grad()

            # forward
            pos_out, ori_out = model(inputs)

            pos_true = gt_t
            ori_true = gt_R

            beta = 500
            ori_out = F.normalize(ori_out, p=2, dim=1)
            ori_true = F.normalize(ori_true, p=2, dim=1)

            loss_pos = F.mse_loss(pos_out, pos_true)
            loss_ori = F.mse_loss(ori_out, ori_true)

            loss = loss_pos + beta * loss_ori

            loss_print = loss.item()
            loss_ori_print = loss_ori.item()
            loss_pos_print = loss_pos.item()

            writer.add_scalar('loss/overall_loss', loss_print, n_iter)
            writer.add_scalar('loss/position_loss', loss_pos_print, n_iter)
            writer.add_scalar('loss/rotation_loss', loss_ori_print, n_iter)
            loss.backward()
            optimizer.step()
            scheduler.step()

            n_iter += 1
            print('{} Loss: total loss {:.3f} / pos loss {:.3f} / ori loss {:.3f}'.format("train", loss_print, loss_pos_print, loss_ori_print))

            if n_iter%val_frequency==0:
                model.eval()
                loss_print = 0
                loss_ori_print = 0
                loss_pos_print =0
                for i, (inputs, gt_t, gt_R) in enumerate(val_data_loader):

                    inputs = inputs.to(device)
                    gt_t = gt_t.to(device)
                    gt_R = gt_R.to(device)

                    # Zero the parameter gradient
                    optimizer.zero_grad()

                    # forward
                    pos_out, ori_out = model(inputs)

                    pos_true = gt_t
                    ori_true = gt_R

                    beta = 1
                    ori_out = F.normalize(ori_out, p=2, dim=1)
                    ori_true = F.normalize(ori_true, p=2, dim=1)

                    loss_pos = F.mse_loss(pos_out, pos_true)
                    loss_ori = F.mse_loss(ori_out, ori_true)

                    loss = loss_pos + beta * loss_ori

                    loss_print += loss.item()
                    loss_ori_print += loss_ori.item()
                    loss_pos_print += loss_pos.item()
                print('{}_epoch{} Loss: total loss {:.3f} / pos loss {:.3f} / ori loss {:.3f}'.format("valid", epoch, loss_print/len(val_data_loader), loss_pos_print/len(val_data_loader), loss_ori_print/len(val_data_loader)))


        save_filename = 'models/%s_net.pth' % (epoch)
        # save_path = os.path.join('models', save_filename)
        torch.save(model.cpu().state_dict(), save_filename)
        if torch.cuda.is_available():
            model.to(device)