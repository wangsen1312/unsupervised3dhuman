from __future__ import print_function, division
import argparse
import sys, os, shutil
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import trimesh
import joblib
import smplx
import open3d as o3d
from pytorch3d.transforms import (axis_angle_to_matrix,
                                  matrix_to_rotation_6d,
                        		      rotation_6d_to_matrix,
                                  matrix_to_quaternion,
                                  quaternion_to_axis_angle)
from src.utils import index_points, farthest_point_sample
from src.Network import point_net_ssg
from src.surfaceem import surface_EM_depth


# parsing argmument
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1,  help='input batch size')
parser.add_argument('--gender', type=str, default="male", help='input male/female/neutral SMPL model')
parser.add_argument('--num_iters', type=int, default=30, help='num of register iters')
parser.add_argument('--gpu_ids', type=int, default=0,  help='choose gpu ids')
parser.add_argument('--restore_path', type=str, default="./pretrained/model_best_depth.pth",  help='pretrained depth model path')
parser.add_argument('--smplmodel_folder', type=str, default="./smpl_models/",  help='pretrained Depth model path')
parser.add_argument('--SMPL_downsample', type=str, default="./smpl_models/SMPL_downsample_index.pkl",  help='downsamople ')
parser.add_argument('--dirs_save', type=str, default="./demo/demo_depth_save/",  help='save directory')
parser.add_argument('--filename', type=str, default="./demo/demo_depth/shortshort_flying_eagle.000075_depth.ply",  help='file for processing')
opt = parser.parse_args()
print(opt)



# Load all Training settings
if torch.cuda.is_available():
    device = torch.device("cuda:" + str(opt.gpu_ids))
else:
    raise ValueError('NO Cuda device detected!')
     
# --------pytorch model and optimizer is the key
model = point_net_ssg(device=device).to(device).eval()
model.load_state_dict(torch.load(opt.restore_path, map_location=device))

optimizer = optim.Adam(model.parameters())
smplmodel = smplx.create(opt.smplmodel_folder, model_type="smpl",
                         gender=opt.gender, ext="pkl").to(device)
                         
# -- intial EM 
# --- load predefined ------
pred_pose = torch.zeros(opt.batchSize, 72).to(device)
pred_betas = torch.zeros(opt.batchSize, 10).to(device)
pred_cam_t = torch.zeros(opt.batchSize, 3).to(device)
trans_back = torch.zeros(opt.batchSize, 3).to(device)

# # #-------------initialize EM -------
loaded_index = joblib.load(opt.SMPL_downsample)
selected_index = loaded_index['downsample_index']

depthEM = surface_EM_depth(smplxmodel=smplmodel,
                           batch_size=opt.batchSize,
                           num_iters=opt.num_iters,
                           selected_index=selected_index,
                           device=device)
                          
os.makedirs(opt.dirs_save, exist_ok=True)

file_name = opt.filename
filename_pure = os.path.splitext(os.path.basename(file_name))[0]
print(filename_pure)


# load mesh and sampling
mesh = trimesh.load(file_name)
point_o = mesh.vertices
pts = torch.from_numpy(point_o).float()
index =  farthest_point_sample(pts.unsqueeze(0), npoint=2048).squeeze()
pts = pts[index]

# move to center
trans = torch.mean(pts, dim=0, keepdim=True)
pts = torch.sub(pts, trans)
point_arr   = torch.transpose(pts, 1, 0)
point_arr =  point_arr.unsqueeze(0).to(device)

point_arr2 = pts.unsqueeze(0).to(device)

# do the inference
with torch.no_grad():
    pred_shape, pred_pose_body, pred_trans, pred_R6D = model(point_arr) #
pred_R6D_3D = quaternion_to_axis_angle(matrix_to_quaternion((rotation_6d_to_matrix(pred_R6D))))

pred_pose[0, 3:] = pred_pose_body.unsqueeze(0).float()
pred_pose[0, :3] = pred_R6D_3D.unsqueeze(0).float()
pred_cam_t[0, :] = pred_trans.unsqueeze(0).float()
trans_back[0, :] = trans.unsqueeze(0).float()

new_opt_vertices, new_opt_joints, new_opt_pose, new_opt_betas, \
new_opt_cam_t =  depthEM(
                         pred_pose.detach(),
                         pred_betas.detach(),   
                         pred_cam_t.detach(),
                         point_arr2
                         )

#save the final results
output = smplmodel(betas=new_opt_betas, global_orient=new_opt_pose[:, :3], body_pose=new_opt_pose[:, 3:],
                   transl=new_opt_cam_t+trans_back, return_verts=True)
mesh = trimesh.Trimesh(vertices=output.vertices.detach().cpu().numpy().squeeze(), faces=smplmodel.faces, process=False)
mesh.export(opt.dirs_save + filename_pure + "_EM.ply")
# also copy the orig files here
shutil.copy(file_name, opt.dirs_save + os.path.basename(file_name)) 

joints3d = output.joints
param = {}
param['joints3d'] = joints3d.detach().cpu().numpy().squeeze()
param['shape'] = new_opt_betas.detach().cpu().numpy()
param['pose'] = new_opt_pose.detach().cpu().numpy()
param['trans'] = new_opt_cam_t.detach().cpu().numpy()
joblib.dump(param, opt.dirs_save + filename_pure + "_EM.pkl", compress=3)

              
