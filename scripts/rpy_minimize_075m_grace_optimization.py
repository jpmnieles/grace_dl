import os
import sys
sys.path.append(os.path.join(os.getcwd(),".."))

import copy
import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pytransform3d
import pytransform3d.camera as pc
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
from pytransform3d.urdf import UrdfTransformManager
from pytransform3d.plot_utils import make_3d_axis, plot_vector
from urdf_parser_py.urdf import URDF
from pytransform3d import rotations as pr

import math
import json
import itertools
from scipy.optimize import minimize, least_squares

import torch
from torch.nn.functional import mse_loss

import time
from datetime import datetime
import cv2.aruco as aruco
from pytorch_kinematics.transforms.rotation_conversions import matrix_to_axis_angle


# Helper Functions
def load_json(filename: str):
    # Construct the absolute path by joining the current directory and relative path
    absolute_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', filename)
    # Load the JSON data
    with open(absolute_path, 'r') as file:
        json_data = json.load(file)    
    return json_data

def xml_to_str(robot):
    # To XML string with filtering
    temp_str = robot.to_xml_string()
    words = temp_str.split()
    words[5] = '>'
    urdf_str = ' '.join(words)
    return urdf_str
        
def active_matrix_from_angle(basis, angle):
    c = torch.cos(angle)  # Shape: Nx1
    s = torch.sin(angle)  # Shape: Nx1
    
    if basis == 0:
        R = torch.zeros((angle.size(0), 3, 3), dtype=angle.dtype, device=angle.device)
        R[:, 0, 0] = 1.0
        R[:, 1, 1] = c.squeeze()  # Shape: N
        R[:, 1, 2] = -s.squeeze()  # Shape: N
        R[:, 2, 1] = s.squeeze()    # Shape: N
        R[:, 2, 2] = c.squeeze()    # Shape: N
    elif basis == 1:
        R = torch.zeros((angle.size(0), 3, 3), dtype=angle.dtype, device=angle.device)
        R[:, 0, 0] = c.squeeze()    # Shape: N
        R[:, 0, 2] = s.squeeze()    # Shape: N
        R[:, 1, 1] = 1.0
        R[:, 2, 0] = -s.squeeze()    # Shape: N
        R[:, 2, 2] = c.squeeze()    # Shape: N
    elif basis == 2:
        R = torch.zeros((angle.size(0), 3, 3), dtype=angle.dtype, device=angle.device)
        R[:, 0, 0] = c.squeeze()    # Shape: N
        R[:, 0, 1] = -s.squeeze()   # Shape: N
        R[:, 1, 0] = s.squeeze()    # Shape: N
        R[:, 1, 1] = c.squeeze()    # Shape: N
        R[:, 2, 2] = 1.0
    else:
        raise ValueError("Basis must be in [0, 1, 2]")
    
    return R

def active_matrix_from_extrinsic_euler_xyz(alpha, beta, gamma):
    # Calculate the rotation matrices
    R_alpha = active_matrix_from_angle(0, alpha)  # Rotation around x-axis
    R_beta = active_matrix_from_angle(1, beta)    # Rotation around y-axis
    R_gamma = active_matrix_from_angle(2, gamma)   # Rotation around z-axis

    # Combine the matrices using the @ operator
    R = R_gamma @ (R_beta @ R_alpha)
    return R

def get_homogeneous_matrix(roll,pitch,yaw,x,y,z):
    rot_mat = active_matrix_from_extrinsic_euler_xyz(roll,pitch,yaw)
    H = torch.zeros((rot_mat.size(0), 4, 4), dtype=dtype, device=device)
    H[:, :3, :3] = rot_mat
    H[:, 0, 3] = x.squeeze()
    H[:, 1, 3] = y.squeeze()
    H[:, 2, 3] = z.squeeze()
    H[:, 3, 3] = 1.0
    return H


# Explicitly telling to use GPU
torch.set_default_device('cuda')
torch.set_default_dtype(d=torch.float64)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float64

# Load Configs
cam_mtxs = load_json('camera_mtx.json')
gaze_ctrs = load_json('calib_params.json')

# Camera Parameters
left_cam_mtx = torch.Tensor(cam_mtxs['left_eye']['camera_matrix'])
left_dist_coef = torch.Tensor(cam_mtxs['left_eye']['distortion_coefficients']).squeeze()
right_cam_mtx = torch.Tensor(cam_mtxs['right_eye']['camera_matrix'])
right_dist_coef = torch.Tensor(cam_mtxs['right_eye']['distortion_coefficients']).squeeze()

# Decision Variables Initial Value

var_dict= {
    # Lower Neck Tilt
    'lnt_roll_b0': 0.1,
    'lnt_roll_b1': 0.1,
    'lnt_roll_b2': 0,
    'lnt_roll_b3': 0,
    'lnt_pitch_b0': 0.1,
    'lnt_pitch_b1': 0.1,
    'lnt_pitch_b2': 0,
    'lnt_pitch_b3': 0,
    'lnt_yaw_b0': 0.1,
    'lnt_yaw_b1': 0.1,
    'lnt_yaw_b2': 0,
    'lnt_yaw_b3': 0,
    'lnt_x_b0': 0.1,
    'lnt_x_b1': 0.1,
    'lnt_x_b2': 0,
    'lnt_x_b3': 0,
    'lnt_x_b0': 0.1,
    'lnt_x_b1': 0.1,
    'lnt_x_b2': 0,
    'lnt_x_b3': 0,
    'lnt_y_b0': 0.1,
    'lnt_y_b1': 0.1,
    'lnt_y_b2': 0,
    'lnt_y_b3': 0,
    'lnt_z_b0': 0.1,
    'lnt_z_b1': 0.1,
    'lnt_z_b2': 0,
    'lnt_z_b3': 0,

    # Lower Neck Pan
    'lnp_roll_b0': 0.1,
    'lnp_roll_b1': 0.1,
    'lnp_roll_b2': 0,
    'lnp_roll_b3': 0,
    'lnp_pitch_b0': 0.1,
    'lnp_pitch_b1': 0.1,
    'lnp_pitch_b2': 0,
    'lnp_pitch_b3': 0,
    'lnp_yaw_b0': 0.1,
    'lnp_yaw_b1': 0.1,
    'lnp_yaw_b2': 0,
    'lnp_yaw_b3': 0,
    'lnp_x_b0': 0.1,
    'lnp_x_b1': 0.1,
    'lnp_x_b2': 0,
    'lnp_x_b3': 0,
    'lnp_y_b0': 0.1,
    'lnp_y_b1': 0.1,
    'lnp_y_b2': 0,
    'lnp_y_b3': 0,
    'lnp_z_b0': 0.1,
    'lnp_z_b1': 0.1,
    'lnp_z_b2': 0,
    'lnp_z_b3': 0,

    # Upper Neck Tilt
    'unt_roll_b0': 0.1,
    'unt_roll_b1': 0.1,
    'unt_roll_b2': 0,
    'unt_roll_b3': 0,
    'unt_pitch_b0': 0.1,
    'unt_pitch_b1': 0.1,
    'unt_pitch_b2': 0,
    'unt_pitch_b3': 0,
    'unt_yaw_b0': 0.1,
    'unt_yaw_b1': 0.1,
    'unt_yaw_b2': 0,
    'unt_yaw_b3': 0,
    'unt_x_b0': 0.1,
    'unt_x_b1': 0.1,
    'unt_x_b2': 0,
    'unt_x_b3': 0,
    'unt_y_b0': 0.1,
    'unt_y_b1': 0.1,
    'unt_y_b2': 0,
    'unt_y_b3': 0,
    'unt_z_b0': 0.1,
    'unt_z_b1': 0.1,
    'unt_z_b2': 0,
    'unt_z_b3': 0,

    # Eyes Tilt
    'et_roll_b0': 0.1,
    'et_roll_b1': 0.1,
    'et_roll_b2': 0,
    'et_roll_b3': 0,
    'et_pitch_b0': 0.1,
    'et_pitch_b1': 0.1,
    'et_pitch_b2': 0,
    'et_pitch_b3': 0,
    'et_yaw_b0': 0.1,
    'et_yaw_b1': 0.1,
    'et_yaw_b2': 0,
    'et_yaw_b3': 0,
    'et_x_b0': 0.1,
    'et_x_b1': 0.1,
    'et_x_b2': 0,
    'et_x_b3': 0,
    'et_y_b0': 0.1,
    'et_y_b1': 0.1,
    'et_y_b2': 0,
    'et_y_b3': 0,
    'et_z_b0': 0.1,
    'et_z_b1': 0.1,
    'et_z_b2': 0,
    'et_z_b3': 0,


    # Left Eye Pan
    'lep_roll_b0': 0.1,
    'lep_roll_b1': 0.1,
    'lep_roll_b2': 0,
    'lep_roll_b3': 0,
    'lep_pitch_b0': 0.1,
    'lep_pitch_b1': 0.1,
    'lep_pitch_b2': 0,
    'lep_pitch_b3': 0,
    'lep_yaw_b0': 0.1,
    'lep_yaw_b1': 0.1,
    'lep_yaw_b2': 0,
    'lep_yaw_b3': 0,
    'lep_x_b0': 0.1,
    'lep_x_b1': 0.1,
    'lep_x_b2': 0,
    'lep_x_b3': 0,
    'lep_y_b0': 0.1,
    'lep_y_b1': 0.1,
    'lep_y_b2': 0,
    'lep_y_b3': 0,
    'lep_z_b0': 0.1,
    'lep_z_b1': 0.1,
    'lep_z_b2': 0,
    'lep_z_b3': 0,
}

# Decision Variables 
var_list = list(var_dict.values())

# Variable Names
var_names_list = list(var_dict.keys())

# Mapping
idx2var = dict(zip(list(range(len(var_names_list))),var_names_list))
var2idx = dict(zip(var_names_list, list(range(len(var_names_list)))))


def objective_function(V, cmd_lnt, cmd_lnp, cmd_unt, cmd_et, cmd_lep,
                      l_rvec, l_tvec, T_home):
    
    # Convert to Tensor
    cmd_lnt = torch.Tensor(cmd_lnt).reshape(-1,1).to(dtype=dtype, device=device)
    cmd_lnp = torch.Tensor(cmd_lnp).reshape(-1,1).to(dtype=dtype, device=device)
    cmd_unt = torch.Tensor(cmd_unt).reshape(-1,1).to(dtype=dtype, device=device)
    cmd_et = torch.Tensor(cmd_et).reshape(-1,1).to(dtype=dtype, device=device)
    cmd_lep = torch.Tensor(cmd_lep).reshape(-1,1).to(dtype=dtype, device=device)

    l_rvec_t = torch.Tensor(l_rvec).to(dtype=dtype, device=device)
    l_tvec_t = torch.Tensor(l_tvec).to(dtype=dtype, device=device)

    ### Joint-to-Motor Polynomial Models
    x = torch.Tensor(cmd_lnt).reshape(-1,1).to(dtype=dtype, device=device)
    lnt_roll = (torch.cat((torch.ones(x.shape[0], 1), x, x.pow(2), x.pow(3)), dim=1) @ torch.Tensor([[V[0]],[V[1]],[V[2]],[V[3]]]).to(dtype=dtype, device=device))
    lnt_pitch = (torch.cat((torch.ones(x.shape[0], 1), x, x.pow(2), x.pow(3)), dim=1) @ torch.Tensor([[V[4]],[V[5]],[V[6]],[V[7]]]).to(dtype=dtype, device=device))
    lnt_yaw = (torch.cat((torch.ones(x.shape[0], 1), x, x.pow(2), x.pow(3)), dim=1) @ torch.Tensor([[V[8]],[V[9]],[V[10]],[V[11]]]).to(dtype=dtype, device=device))
    lnt_x = (torch.cat((torch.ones(x.shape[0], 1), x, x.pow(2), x.pow(3)), dim=1) @ torch.Tensor([[V[12]],[V[13]],[V[14]],[V[15]]]).to(dtype=dtype, device=device))
    lnt_y = (torch.cat((torch.ones(x.shape[0], 1), x, x.pow(2), x.pow(3)), dim=1) @ torch.Tensor([[V[16]],[V[17]],[V[18]],[V[19]]]).to(dtype=dtype, device=device))
    lnt_z = (torch.cat((torch.ones(x.shape[0], 1), x, x.pow(2), x.pow(3)), dim=1) @ torch.Tensor([[V[20]],[V[21]],[V[22]],[V[23]]]).to(dtype=dtype, device=device))

    x = torch.Tensor(cmd_lnp).reshape(-1,1).to(dtype=dtype, device=device)
    lnp_roll = (torch.cat((torch.ones(x.shape[0], 1), x, x.pow(2), x.pow(3)), dim=1) @ torch.Tensor([[V[24]],[V[25]],[V[26]],[V[27]]]).to(dtype=dtype, device=device))
    lnp_pitch = (torch.cat((torch.ones(x.shape[0], 1), x, x.pow(2), x.pow(3)), dim=1) @ torch.Tensor([[V[28]],[V[29]],[V[30]],[V[31]]]).to(dtype=dtype, device=device))
    lnp_yaw = (torch.cat((torch.ones(x.shape[0], 1), x, x.pow(2), x.pow(3)), dim=1) @ torch.Tensor([[V[32]],[V[33]],[V[34]],[V[35]]]).to(dtype=dtype, device=device))
    lnp_x = (torch.cat((torch.ones(x.shape[0], 1), x, x.pow(2), x.pow(3)), dim=1) @ torch.Tensor([[V[36]],[V[37]],[V[38]],[V[39]]]).to(dtype=dtype, device=device))
    lnp_y = (torch.cat((torch.ones(x.shape[0], 1), x, x.pow(2), x.pow(3)), dim=1) @ torch.Tensor([[V[40]],[V[41]],[V[42]],[V[43]]]).to(dtype=dtype, device=device))
    lnp_z = (torch.cat((torch.ones(x.shape[0], 1), x, x.pow(2), x.pow(3)), dim=1) @ torch.Tensor([[V[44]],[V[45]],[V[46]],[V[47]]]).to(dtype=dtype, device=device))

    x = torch.Tensor(cmd_unt).reshape(-1,1).to(dtype=dtype, device=device)
    unt_roll = (torch.cat((torch.ones(x.shape[0], 1), x, x.pow(2), x.pow(3)), dim=1) @ torch.Tensor([[V[48]],[V[49]],[V[50]],[V[51]]]).to(dtype=dtype, device=device))
    unt_pitch = (torch.cat((torch.ones(x.shape[0], 1), x, x.pow(2), x.pow(3)), dim=1) @ torch.Tensor([[V[52]],[V[53]],[V[54]],[V[55]]]).to(dtype=dtype, device=device))
    unt_yaw = (torch.cat((torch.ones(x.shape[0], 1), x, x.pow(2), x.pow(3)), dim=1) @ torch.Tensor([[V[56]],[V[57]],[V[58]],[V[59]]]).to(dtype=dtype, device=device))
    unt_x = (torch.cat((torch.ones(x.shape[0], 1), x, x.pow(2), x.pow(3)), dim=1) @ torch.Tensor([[V[60]],[V[61]],[V[62]],[V[63]]]).to(dtype=dtype, device=device))
    unt_y = (torch.cat((torch.ones(x.shape[0], 1), x, x.pow(2), x.pow(3)), dim=1) @ torch.Tensor([[V[64]],[V[65]],[V[66]],[V[67]]]).to(dtype=dtype, device=device))
    unt_z = (torch.cat((torch.ones(x.shape[0], 1), x, x.pow(2), x.pow(3)), dim=1) @ torch.Tensor([[V[68]],[V[69]],[V[70]],[V[71]]]).to(dtype=dtype, device=device))

    x = torch.Tensor(cmd_et).reshape(-1,1).to(dtype=dtype, device=device)
    et_roll = (torch.cat((torch.ones(x.shape[0], 1), x, x.pow(2), x.pow(3)), dim=1) @ torch.Tensor([[V[72]],[V[73]],[V[74]],[V[75]]]).to(dtype=dtype, device=device))
    et_pitch = (torch.cat((torch.ones(x.shape[0], 1), x, x.pow(2), x.pow(3)), dim=1) @ torch.Tensor([[V[76]],[V[77]],[V[78]],[V[79]]]).to(dtype=dtype, device=device))
    et_yaw = (torch.cat((torch.ones(x.shape[0], 1), x, x.pow(2), x.pow(3)), dim=1) @ torch.Tensor([[V[80]],[V[81]],[V[82]],[V[83]]]).to(dtype=dtype, device=device))
    et_x = (torch.cat((torch.ones(x.shape[0], 1), x, x.pow(2), x.pow(3)), dim=1) @ torch.Tensor([[V[84]],[V[85]],[V[86]],[V[87]]]).to(dtype=dtype, device=device))
    et_y = (torch.cat((torch.ones(x.shape[0], 1), x, x.pow(2), x.pow(3)), dim=1) @ torch.Tensor([[V[88]],[V[89]],[V[90]],[V[91]]]).to(dtype=dtype, device=device))
    et_z = (torch.cat((torch.ones(x.shape[0], 1), x, x.pow(2), x.pow(3)), dim=1) @ torch.Tensor([[V[92]],[V[93]],[V[94]],[V[95]]]).to(dtype=dtype, device=device))

    x = torch.Tensor(cmd_lep).reshape(-1,1).to(dtype=dtype, device=device)
    lep_roll = (torch.cat((torch.ones(x.shape[0], 1), x, x.pow(2), x.pow(3)), dim=1) @ torch.Tensor([[V[96]],[V[97]],[V[98]],[V[99]]]).to(dtype=dtype, device=device))
    lep_pitch = (torch.cat((torch.ones(x.shape[0], 1), x, x.pow(2), x.pow(3)), dim=1) @ torch.Tensor([[V[100]],[V[101]],[V[102]],[V[103]]]).to(dtype=dtype, device=device))
    lep_yaw = (torch.cat((torch.ones(x.shape[0], 1), x, x.pow(2), x.pow(3)), dim=1) @ torch.Tensor([[V[104]],[V[105]],[V[106]],[V[107]]]).to(dtype=dtype, device=device))
    lep_x = (torch.cat((torch.ones(x.shape[0], 1), x, x.pow(2), x.pow(3)), dim=1) @ torch.Tensor([[V[108]],[V[109]],[V[110]],[V[111]]]).to(dtype=dtype, device=device))
    lep_y = (torch.cat((torch.ones(x.shape[0], 1), x, x.pow(2), x.pow(3)), dim=1) @ torch.Tensor([[V[112]],[V[113]],[V[114]],[V[115]]]).to(dtype=dtype, device=device))
    lep_z = (torch.cat((torch.ones(x.shape[0], 1), x, x.pow(2), x.pow(3)), dim=1) @ torch.Tensor([[V[116]],[V[117]],[V[118]],[V[119]]]).to(dtype=dtype, device=device))
    
    # Home Homogeneous
    T_temp = torch.Tensor(T_home).to(dtype=dtype, device=device)
    T_0 = torch.stack([T_temp] * cmd_lnt.size(0))

    # Homogeneous Matrices
    T_lnt = get_homogeneous_matrix(lnt_roll,lnt_pitch,lnt_yaw,lnt_x,lnt_y,lnt_z)
    T_lnp = get_homogeneous_matrix(lnp_roll,lnp_pitch,lnp_yaw,lnp_x,lnp_y,lnp_z)
    T_unt = get_homogeneous_matrix(unt_roll,unt_pitch,unt_yaw,unt_x,unt_y,unt_z)
    T_et = get_homogeneous_matrix(et_roll,et_pitch,et_yaw,et_x,et_y,et_z)
    T_lep = get_homogeneous_matrix(lep_roll,lep_pitch,lep_yaw,lep_x,lep_y,lep_z)

    # Final Matrix
    T_final = T_0 @ T_lnt @ T_lnp @ T_unt @ T_et @ T_lep
    T_final

    # Predicted
    pred_l_rvec = matrix_to_axis_angle(T_final[:,:3,:3])
    pred_l_tvec = T_final[:,:3,3]

    # Min-max normalization
    min_val = min([pred_l_tvec.min(),l_tvec_t.min()])
    max_val = max([pred_l_tvec.max(),l_tvec_t.max()])

    # Normalized Tvec
    norm_l_tvec_t = (l_rvec_t - min_val) / (max_val - min_val)
    norm_pred_l_tvec = (pred_l_tvec - min_val) / (max_val - min_val)

    # Loss
    residuals = (mse_loss(pred_l_rvec, l_rvec_t) + mse_loss(norm_l_tvec_t, norm_pred_l_tvec)).cpu().item()
    residuals

    # Return Loss
    print(residuals)
    return residuals


def main(data_dir="final", csv_fn="241003_075m_grace_dataset.csv"):
    # Load Dataset CSV
    csv_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","data", data_dir, csv_fn)
    df = pd.read_csv(csv_file)
    
    ## Home df
    home_df = df[(df['cmd_theta_lower_neck_tilt']==0) 
                & (df['cmd_theta_lower_neck_pan']==0) 
                & (df['cmd_theta_upper_neck_tilt']==0)
                & (df['cmd_theta_left_eye_pan']==0)
                & (df['cmd_theta_right_eye_pan']==0)
                & (df['cmd_theta_eyes_tilt']==0)]
    home_df = home_df.reset_index(drop=True)
    # Rvec and Tvec
    home_l_rvec = np.array([home_df['l_rvec_0'],home_df['l_rvec_1'],home_df['l_rvec_2']]).flatten()
    home_l_tvec = np.array([home_df['l_tvec_0'],home_df['l_tvec_1'],home_df['l_tvec_2']]).flatten()
    temp_rmat,_ = cv2.Rodrigues(home_l_rvec)
    T_home = np.eye(4)
    T_home[:3,:3] = temp_rmat.squeeze()
    T_home[:3,3] = home_l_tvec
    T_home_inv = np.linalg.inv(T_home)

    ## Relative to Left Home Position
    T_clprime_list = []
    l_0lprime_pose_list = []
    l_0lprime_roll_list = []
    l_0lprime_pitch_list = []
    l_0lprime_yaw_list = []
    l_0lprime_x_list = []
    l_0lprime_y_list = []
    l_0lprime_z_list = []
    for i in range(len(df)):
        # Getting Homogeneous Matrix
        l_rvec = np.array([df['l_rvec_0'][i],df['l_rvec_1'][i],df['l_rvec_2'][i]])
        l_tvec = np.array([df['l_tvec_0'][i],df['l_tvec_1'][i],df['l_tvec_2'][i]])
        l_rmat,_ = cv2.Rodrigues(l_rvec)
        T_clprime = np.eye(4)
        T_clprime[:3,:3] = l_rmat.squeeze()
        T_clprime[:3,3] = l_tvec.flatten()
        T_clprime_list.append(T_clprime)

        # Getting Relative Rotation
        T_0lprime = T_home_inv @ T_clprime
        roll,pitch,yaw = pr.extrinsic_euler_xyz_from_active_matrix(T_0lprime[:3,:3])
        x,y,z = T_0lprime[:3,3].flatten()
        l_0lprime_x_list.append(x)
        l_0lprime_y_list.append(y)
        l_0lprime_z_list.append(z)
        l_0lprime_roll_list.append(roll)
        l_0lprime_pitch_list.append(pitch)
        l_0lprime_yaw_list.append(yaw)
        l_0lprime_pose_list.append(T_0lprime)

    # Saving to Dataframe
    df['T_clprime'] = T_clprime_list
    df['l_roll'] = l_0lprime_roll_list
    df['l_pitch'] = l_0lprime_pitch_list
    df['l_yaw'] = l_0lprime_yaw_list
    df['l_x'] = l_0lprime_x_list
    df['l_y'] = l_0lprime_y_list
    df['l_z'] = l_0lprime_z_list
    df['T_0lprime'] = l_0lprime_pose_list

    # Removed home position
    temp_df = df[~((df['cmd_theta_lower_neck_tilt']==0) 
                & (df['cmd_theta_lower_neck_pan']==0) 
                & (df['cmd_theta_upper_neck_tilt']==0)
                & (df['cmd_theta_left_eye_pan']==0)
                & (df['cmd_theta_right_eye_pan']==0)
                & (df['cmd_theta_eyes_tilt']==0))]

    # Convert DF Degrees Column to Radians
    data_df = temp_df.copy()
    data_df['cmd_theta_lower_neck_pan'] = np.radians(temp_df['cmd_theta_lower_neck_pan'].values)
    data_df['cmd_theta_lower_neck_tilt'] = np.radians(temp_df['cmd_theta_lower_neck_tilt'].values)
    data_df['cmd_theta_upper_neck_tilt'] = np.radians(temp_df['cmd_theta_upper_neck_tilt'].values)
    data_df['cmd_theta_left_eye_pan'] = np.radians(temp_df['cmd_theta_left_eye_pan'].values)
    data_df['cmd_theta_right_eye_pan'] = np.radians(temp_df['cmd_theta_right_eye_pan'].values)
    data_df['cmd_theta_eyes_tilt'] = np.radians(temp_df['cmd_theta_eyes_tilt'].values)
    data_df
    
    # Input
    V = np.array(var_list)
    cmd_lnt = data_df['cmd_theta_lower_neck_tilt'].to_numpy()
    cmd_lnp = data_df['cmd_theta_lower_neck_pan'].to_numpy()
    cmd_unt = data_df['cmd_theta_upper_neck_tilt'].to_numpy()
    cmd_et = data_df['cmd_theta_eyes_tilt'].to_numpy()
    cmd_lep = data_df['cmd_theta_left_eye_pan'].to_numpy()
    cmd_rep = data_df['cmd_theta_right_eye_pan'].to_numpy()

    l_rvec = np.concatenate([data_df['l_rvec_0'].values.reshape(-1,1),
                             data_df['l_rvec_1'].values.reshape(-1,1),
                             data_df['l_rvec_2'].values.reshape(-1,1)], axis=1)
    l_tvec = np.concatenate([data_df['l_tvec_0'].values.reshape(-1,1),
                             data_df['l_tvec_1'].values.reshape(-1,1),
                             data_df['l_tvec_2'].values.reshape(-1,1)], axis=1)
    r_rvec = np.concatenate([data_df['r_rvec_0'].values.reshape(-1,1),
                             data_df['r_rvec_1'].values.reshape(-1,1),
                             data_df['r_rvec_2'].values.reshape(-1,1)], axis=1)
    r_tvec = np.concatenate([data_df['r_tvec_0'].values.reshape(-1,1),
                             data_df['r_tvec_1'].values.reshape(-1,1),
                             data_df['r_tvec_2'].values.reshape(-1,1)], axis=1)

    # Test Objective Function
    opt = minimize(objective_function, V, args=(cmd_lnt, cmd_lnp, cmd_unt, cmd_et, cmd_lep,
                                                 l_rvec, l_tvec, T_home), 
                                                 method="Powell", 
                                                 options={"disp":True},                     
                                                )
    
    
    print(dict(zip(var_names_list, opt.x)))

    # Saving URDF Results
    V = opt.x

    # Print and save the results to csv
    res_df = pd.DataFrame({
        'initial': var_list,
        'learned': V,
    })
    res_df.index = var_names_list
    print(res_df)
    res_fn = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    res_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..','results', res_fn+'_rpy_075m_grace_results.csv')
    res_df.to_csv(res_path)
    print(f"CSV results saved to {res_path}")


if __name__ == '__main__':
    start = time.time()
    main("final","241003_075m_grace_dataset.csv")
    end = time.time()
    print('Elapsed Time (sec):',end-start)