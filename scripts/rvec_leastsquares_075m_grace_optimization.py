import os
import sys
sys.path.append(os.path.join(os.getcwd(),".."))

import time
import math
import json
import glob
from datetime import datetime

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize, least_squares
import matplotlib.pyplot as plt

import cv2
import torch
from torch.nn.functional import mse_loss
from pytorch_kinematics.transforms.rotation_conversions import matrix_to_axis_angle, axis_angle_to_matrix

import pytorch_kinematics as pk
from pytransform3d.urdf import UrdfTransformManager
from urdf_parser_py.urdf import URDF


# Helper Functions
def load_json(filename: str):
    # Construct the absolute path by joining the current directory and relative path
    absolute_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', filename)
    # Load the JSON data
    with open(absolute_path, 'r') as file:
        json_data = json.load(file)    
    return json_data


# Explicitly telling to use GPU
torch.set_default_device('cuda')
torch.set_default_dtype(d=torch.float64)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float64

# Load Configs
cam_mtxs = load_json('camera_mtx.json')
gaze_ctrs = load_json('calib_params.json')
urdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","urdf","chest_grace_cam_true.urdf")
robot = URDF.from_xml_file(urdf_path)

# Load Dataset CSV
data_dir = "final"
csv_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","data", data_dir, "240927_075m_grace_dataset.csv")
temp_df = pd.read_csv(csv_file)

# Camera Parameters
left_cam_mtx = torch.Tensor(cam_mtxs['left_eye']['camera_matrix'])
left_dist_coef = torch.Tensor(cam_mtxs['left_eye']['distortion_coefficients']).squeeze()
right_cam_mtx = torch.Tensor(cam_mtxs['right_eye']['camera_matrix'])
right_dist_coef = torch.Tensor(cam_mtxs['right_eye']['distortion_coefficients']).squeeze()

# Decision Variables Initial Value

var_dict= {
    'neck_pitch_polyfit_b0': 0.1, 
    'neck_yaw_polyfit_b0': 0.1,
    'head_pitch_polyfit_b0': 0.1,
    'eyes_pitch_polyfit_b0': 0.1,
    'lefteye_yaw_polyfit_b0': 0.1,
    'righteye_yaw_polyfit_b0': 0.1,

    'neck_pitch_polyfit_b1': 0.5, 
    'neck_yaw_polyfit_b1': 0.5,
    'head_pitch_polyfit_b1': 0.5,
    'eyes_pitch_polyfit_b1': 0.4,
    'lefteye_yaw_polyfit_b1': 1.6,
    'righteye_yaw_polyfit_b1': 1.7,

    'neck_pitch_rot_r': 0,

    'torso_origin_x': 0.0325,
    'torso_origin_y': -0.05692,
    'torso_origin_z': -0.12234,
    'head_pitch_origin_z': 0.13172,
    
    'eyes_pitch_origin_x': 0.08492,
    'eyes_pitch_origin_z': 0.05186,
    'lefteye_yaw_origin_y': 0.02895,
    # 'righteye_yaw_origin_y': -0.02895,
    'lefteye_cam_origin_x': 0.015,
    'righteye_cam_origin_x': 0.015,

    'lefteye_cam_rot_r': -1.5362233798559841,
    'lefteye_cam_rot_y': -1.3816620298291191,

    'righteye_cam_rot_r': -1.5152648840678353,
    'righteye_cam_rot_y': -1.8252206094843146,

}

# Decision Variables 
var_list = list(var_dict.values())

# Variable Names
var_names_list = list(var_dict.keys())

# Mapping
idx2var = dict(zip(list(range(len(var_names_list))),var_names_list))
var2idx = dict(zip(var_names_list, list(range(len(var_names_list)))))


def projectPoints(pts_3d, ext_mtx, cam_mtx, dist_coef):
    # Apply extrinsic transformation (rotation + translation)
    pts_3d_transformed = (ext_mtx @ pts_3d).reshape(-1,4).T
    
    # Convert back from homogeneous to 2D coordinates
    pts_2d_prime = pts_3d_transformed[:2, :] / pts_3d_transformed[2, :]
    x_prime = pts_2d_prime[0]
    y_prime = pts_2d_prime[1]
    
    # Distortion Parameters Calculation
    k1, k2, p1, p2, k3 = dist_coef
    
    r2 = x_prime**2 + y_prime**2
    r4 = r2**2
    r6 = r2*r4
    a1 = 2*x_prime*y_prime
    a2 = r2 + 2*x_prime**2
    a3 = r2 + 2*y_prime**2
    
    # Radial distortion
    radial_distortion = 1 + k1*r2 + k2*r4 + k3*r6
    
    # Tangential distortion
    tangential_distortion_x = p1*a1 + p2*a2
    tangential_distortion_y = p1*a3 + p2*a1
    
    # Apply distortion
    x_double_prime = x_prime*radial_distortion + tangential_distortion_x
    y_double_prime = y_prime* radial_distortion + tangential_distortion_y
    
    # Intrinsic Camera Matrix
    fx = cam_mtx[0,0]
    cx = cam_mtx[0,2]
    fy = cam_mtx[1,1]
    cy = cam_mtx[1,2]
    
    u = x_double_prime*fx + cx
    v = y_double_prime*fy + cy

    return u,v

def xml_to_str(robot):
    # To XML string with filtering
    temp_str = robot.to_xml_string()
    words = temp_str.split()
    words[5] = '>'
    urdf_str = ' '.join(words)
    return urdf_str

def min_max_scaler(tensor, tensor_min, tensor_max):  
    # Apply Min-Max scaling
    scaled_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
    
    return scaled_tensor


def objective_function(V, x_c_l, y_c_l, z_c_l, x_c_r, y_c_r, z_c_r,
                      cmd_lnt, cmd_lnp, cmd_unt, cmd_et, cmd_lep, cmd_rep,
                      l_rvec, l_tvec, r_rvec, r_tvec):
    
    # Convert to Tensor
    x_c_l = torch.Tensor(x_c_l).reshape(-1,1).to(dtype=dtype, device=device)
    y_c_l = torch.Tensor(y_c_l).reshape(-1,1).to(dtype=dtype, device=device)
    z_c_l = torch.Tensor(z_c_l).reshape(-1,1).to(dtype=dtype, device=device)
    x_c_r = torch.Tensor(x_c_r).reshape(-1,1).to(dtype=dtype, device=device)
    y_c_r = torch.Tensor(y_c_r).reshape(-1,1).to(dtype=dtype, device=device)
    z_c_r = torch.Tensor(z_c_r).reshape(-1,1).to(dtype=dtype, device=device)
    cmd_lnt = torch.Tensor(cmd_lnt).reshape(-1,1).to(dtype=dtype, device=device)
    cmd_lnp = torch.Tensor(cmd_lnp).reshape(-1,1).to(dtype=dtype, device=device)
    cmd_unt = torch.Tensor(cmd_unt).reshape(-1,1).to(dtype=dtype, device=device)
    cmd_et = torch.Tensor(cmd_et).reshape(-1,1).to(dtype=dtype, device=device)
    cmd_lep = torch.Tensor(cmd_lep).reshape(-1,1).to(dtype=dtype, device=device)
    cmd_rep = torch.Tensor(cmd_rep).reshape(-1,1).to(dtype=dtype, device=device)

    l_rvec_t = torch.Tensor(l_rvec).to(dtype=dtype, device=device)
    l_tvec_t = torch.Tensor(l_tvec).to(dtype=dtype, device=device)
    r_rvec_t = torch.Tensor(r_rvec).to(dtype=dtype, device=device)
    r_tvec_t = torch.Tensor(r_tvec).to(dtype=dtype, device=device)

    #Joint-to-Motor Polynomial Models
    # neck_pitch = cmd_lnt @ torch.Tensor([V[var2idx['neck_pitch_polyfit_b1']]]).to(dtype=dtype, device=device)
    # neck_yaw = cmd_lnp @ torch.Tensor([V[var2idx['neck_yaw_polyfit_b1']]]).to(dtype=dtype, device=device)
    # head_pitch = cmd_unt @ torch.Tensor([V[var2idx['head_pitch_polyfit_b1']]]).to(dtype=dtype, device=device)
    # eyes_pitch = cmd_et @ torch.Tensor([V[var2idx['eyes_pitch_polyfit_b1']]]).to(dtype=dtype, device=device)
    # lefteye_yaw = cmd_lep @ torch.Tensor([V[var2idx['lefteye_yaw_polyfit_b1']]]).to(dtype=dtype, device=device)
    # righteye_yaw = cmd_rep @ torch.Tensor([V[var2idx['righteye_yaw_polyfit_b1']]]).to(dtype=dtype, device=device)

    neck_pitch = V[var2idx['neck_pitch_polyfit_b0']] + cmd_lnt @ torch.Tensor([V[var2idx['neck_pitch_polyfit_b1']]]).to(dtype=dtype, device=device)
    neck_yaw = V[var2idx['neck_yaw_polyfit_b0']] + cmd_lnp @ torch.Tensor([V[var2idx['neck_yaw_polyfit_b1']]]).to(dtype=dtype, device=device)
    head_pitch = V[var2idx['head_pitch_polyfit_b0']] + cmd_unt @ torch.Tensor([V[var2idx['head_pitch_polyfit_b1']]]).to(dtype=dtype, device=device)
    eyes_pitch = V[var2idx['eyes_pitch_polyfit_b0']] + cmd_et @ torch.Tensor([V[var2idx['eyes_pitch_polyfit_b1']]]).to(dtype=dtype, device=device)
    lefteye_yaw = V[var2idx['lefteye_yaw_polyfit_b0']] + cmd_lep @ torch.Tensor([V[var2idx['lefteye_yaw_polyfit_b1']]]).to(dtype=dtype, device=device)
    righteye_yaw = V[var2idx['righteye_yaw_polyfit_b0']] + cmd_rep @ torch.Tensor([V[var2idx['righteye_yaw_polyfit_b1']]]).to(dtype=dtype, device=device)

    # neck_pitch = cmd_lnt @ torch.Tensor([0.5]).to(dtype=dtype, device=device)
    # neck_yaw = cmd_lnp @ torch.Tensor([0.5]).to(dtype=dtype, device=device)
    # head_pitch = cmd_unt @ torch.Tensor([0.5]).to(dtype=dtype, device=device)
    # eyes_pitch = cmd_et @ torch.Tensor([0.4]).to(dtype=dtype, device=device)
    # lefteye_yaw = cmd_lep @ torch.Tensor([1.6]).to(dtype=dtype, device=device)
    # righteye_yaw = cmd_rep @ torch.Tensor([1.7]).to(dtype=dtype, device=device)


    # URDF Variable Assignment
    for joint in robot.joints:
        if joint.name == 'torso':
            joint.origin.position[0] = V[var2idx['torso_origin_x']]
            joint.origin.position[1] = V[var2idx['torso_origin_y']]
            joint.origin.position[2] = V[var2idx['torso_origin_z']]
        elif joint.name == 'neck_pitch':
            joint.origin.rotation[0] = V[var2idx['neck_pitch_rot_r']]
        elif joint.name == 'head_pitch':
            joint.origin.position[2] = V[var2idx['head_pitch_origin_z']]
            # joint.origin.rotation[0] = V[29]
            # joint.origin.rotation[2] = V[30]
        elif joint.name == 'eyes_pitch':
            joint.origin.position[0] = V[var2idx['eyes_pitch_origin_x']]
            joint.origin.position[2] = V[var2idx['eyes_pitch_origin_z']]
            # joint.origin.rotation[0] = V[33]
            # joint.origin.rotation[2] = V[34]
        elif joint.name == 'lefteye_yaw':
            joint.origin.position[1] = V[var2idx['lefteye_yaw_origin_y']]
        elif joint.name == 'righteye_yaw':
            joint.origin.position[1] = -V[var2idx['lefteye_yaw_origin_y']]
        elif joint.name == 'lefteye_cam':
            joint.origin.position[0] = V[var2idx['lefteye_cam_origin_x']]
            joint.origin.rotation[0] = V[var2idx['lefteye_cam_rot_r']]
            # joint.origin.rotation[1] = V[var2idx['lefteye_cam_rot_p']]
            joint.origin.rotation[2] = V[var2idx['lefteye_cam_rot_y']]
        elif joint.name == 'righteye_cam':
            joint.origin.position[0] = V[var2idx['righteye_cam_origin_x']]
            joint.origin.rotation[0] = V[var2idx['righteye_cam_rot_r']]
            # joint.origin.rotation[1] = V[var2idx['righteye_cam_rot_p']]
            joint.origin.rotation[2] = V[var2idx['righteye_cam_rot_y']]


    # # # URDF Variable Assignment
    # for joint in robot.joints:
    #     if joint.name == 'lefteye_cam':
    #         joint.origin.position[0] = 0.015 + 0
    #         joint.origin.rotation[0] = -1.5708 + 0.03457662014401592  # For Y, Downwards is + because of Camera Opencv Coordinate System
    #         joint.origin.rotation[1] = 0 + 0
    #         joint.origin.rotation[2] = -1.5708 + 0.18913797017088096  # Orientation Offset: Right Hand Rule (Positive CCW)
    #     elif joint.name == 'righteye_cam':
    #         joint.origin.position[0] = 0.015 + 0
    #         joint.origin.rotation[0] = -1.5708 + 0.05553511593216458  # For Y, Downwards is + because of Camera Opencv Coordinate System
    #         joint.origin.rotation[1] = 0 + 0
    #         joint.origin.rotation[2] = -1.5708 + -0.2544206094843146  # Orientation Offset: Right Hand Rule (Positive CCW)
    
    # XML to String
    urdf_str = xml_to_str(robot)
    
    # Kinematic Chain
    chain = pk.build_chain_from_urdf(urdf_str)
    chain.to(dtype=dtype, device=device)
    
    # Specifying Joint Angles (radians)
    joint_cmd = torch.cat((neck_pitch.reshape(-1,1), neck_yaw.reshape(-1,1), head_pitch.reshape(-1,1), eyes_pitch.reshape(-1,1), lefteye_yaw.reshape(-1,1), righteye_yaw.reshape(-1,1)), dim=1)


    # Forward Kinematics
    ret = chain.forward_kinematics(joint_cmd)

    # Realsense to Left Eye Camera (Pytorch) with Points
    T_clprime = torch.linalg.inv(ret['leftcamera'].get_matrix()) @ ret['realsense'].get_matrix()
    
    # Realsense to Right Eye Camera (Pytorch) with Points
    T_crprime = torch.linalg.inv(ret['rightcamera'].get_matrix()) @ ret['realsense'].get_matrix()
    
    # # Chest Camera OpenCV Orientation 3D Projection Points
    # left_chest_pts = torch.cat((x_c_l, y_c_l, z_c_l,torch.ones(x_c_l.shape[0], 1)), dim=1).reshape(-1,4,1)
    # right_chest_pts = torch.cat((x_c_r, y_c_r, z_c_r,torch.ones(x_c_r.shape[0], 1)), dim=1).reshape(-1,4,1)
    
    # # Points Projection
    # u_left, v_left = projectPoints(left_chest_pts, T_clprime, left_cam_mtx, left_dist_coef)   
    # u_right, v_right = projectPoints(right_chest_pts, T_crprime, right_cam_mtx, right_dist_coef)

    # # Clamping

    # u_min_val = 0
    # u_max_val = 639 
    # v_min_val = 0
    # v_max_val = 479
    # # u_left = torch.clamp(u_left, min=u_min_val, max=u_max_val)
    # # v_left = torch.clamp(v_left, min=v_min_val, max=v_max_val)
    # # u_right = torch.clamp(u_right, min=u_min_val, max=u_max_val)
    # # v_right = torch.clamp(v_right, min=v_min_val, max=v_max_val)
    
    # # Scaling
    # u_left = min_max_scaler(u_left, u_min_val, u_max_val)
    # v_left = min_max_scaler(v_left, v_min_val, v_max_val)
    # u_right = min_max_scaler(u_right, u_min_val, u_max_val)
    # v_right = min_max_scaler(v_right, v_min_val, v_max_val)
    
    # # True Value
    # true_u_left = gaze_ctrs['left_eye']['x_center']
    # true_v_left = gaze_ctrs['left_eye']['y_center']
    # true_u_right = gaze_ctrs['right_eye']['x_center']
    # true_v_right = gaze_ctrs['right_eye']['y_center']
    # true_u_left = min_max_scaler(gaze_ctrs['left_eye']['x_center'], u_min_val, u_max_val)
    # true_v_left = min_max_scaler(gaze_ctrs['left_eye']['y_center'], v_min_val, v_max_val)
    # true_u_right = min_max_scaler(gaze_ctrs['right_eye']['x_center'], u_min_val, u_max_val)
    # true_v_right = min_max_scaler(gaze_ctrs['right_eye']['y_center'], v_min_val, v_max_val)

    
    # # Loss
    # residuals = torch.cat((((u_left-true_u_left)**2).reshape(-1,1),((v_left-true_v_left)**2).reshape(-1,1),((u_right-true_u_right)**2).reshape(-1,1),((v_right-true_v_right)**2).reshape(-1,1)), dim=1).cpu().numpy()
    # res_sum = np.sum(residuals)

    # Rvec and Tvec
    out_l_rvec = matrix_to_axis_angle(T_clprime[:,:3,:3])
    out_l_tvec = T_clprime[:,:3,3]
    out_r_rvec = matrix_to_axis_angle(T_crprime[:,:3,:3])
    out_r_tvec = T_crprime[:,:3,3]

    # Loss
    # residuals = (mse_loss(out_l_rvec, l_rvec_t, reduction='sum') + mse_loss(out_l_tvec, l_tvec_t, reduction='sum')
    #              + mse_loss(out_r_rvec, r_rvec_t, reduction='sum') + mse_loss(out_r_tvec, r_tvec_t, reduction='sum')).cpu().item()
    
    residuals = torch.cat((((out_l_rvec - l_rvec_t)**2).reshape(-1,3),
                           ((out_l_tvec - l_tvec_t)**2).reshape(-1,3),
                           ((out_r_rvec - r_rvec_t)**2).reshape(-1,3),
                           ((out_r_tvec - r_tvec_t)**2).reshape(-1,3)), dim=1).cpu().numpy()
    res_sum = np.sum(residuals, axis=1)

    # Return Loss
    return res_sum


def main():
    # Sample 1
    # small_df = temp_df.sample(n=10000, random_state=1).reset_index(drop=True)
    small_df = temp_df.copy()
    # small_df = temp_df[(temp_df['cmd_theta_lower_neck_pan']==0)
    #                    &(temp_df['cmd_theta_lower_neck_tilt']==0)
    #                    &(temp_df['cmd_theta_upper_neck_tilt']==0)
    #                    &(temp_df['cmd_theta_left_eye_pan']==0)
    #                    &(temp_df['cmd_theta_right_eye_pan']==0)].reset_index(drop=True)

    # Convert DF Degrees Column to Radians
    data_df = small_df.copy()
    data_df['cmd_theta_lower_neck_pan'] = np.radians(data_df['cmd_theta_lower_neck_pan'].values)
    data_df['cmd_theta_lower_neck_tilt'] = np.radians(data_df['cmd_theta_lower_neck_tilt'].values)
    data_df['cmd_theta_upper_neck_tilt'] = np.radians(data_df['cmd_theta_upper_neck_tilt'].values)
    data_df['cmd_theta_left_eye_pan'] = np.radians(data_df['cmd_theta_left_eye_pan'].values)
    data_df['cmd_theta_right_eye_pan'] = np.radians(data_df['cmd_theta_right_eye_pan'].values)
    data_df['cmd_theta_eyes_tilt'] = np.radians(data_df['cmd_theta_eyes_tilt'].values)
    
    # Input
    V = np.array(var_list)
    x_c_l = data_df['x_c_l'].to_numpy()
    y_c_l = data_df['y_c_l'].to_numpy()
    z_c_l = data_df['z_c_l'].to_numpy()
    x_c_r = data_df['x_c_r'].to_numpy()
    y_c_r = data_df['y_c_r'].to_numpy()
    z_c_r = data_df['z_c_r'].to_numpy()
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
    opt = least_squares(objective_function, V, verbose=2, args=(x_c_l, y_c_l, z_c_l, x_c_r, y_c_r, z_c_r,
                                                 cmd_lnt, cmd_lnp, cmd_unt, cmd_et, cmd_lep, cmd_rep,
                                                 l_rvec, l_tvec, r_rvec, r_tvec),
                                                 bounds=([-1,-1,-1,-1,-1,-1,
                                                          0,0,0,0,0,0,
                                                          -1,
                                                          0,-1,-1,0,
                                                          0,0,
                                                          0,
                                                          0,0,
                                                          -3.1416,-3.1416,-3.1416,-3.1416],
                                                         [1,1,1,1,1,1,
                                                          4,4,4,4,4,4,
                                                          1,
                                                          1,0,0,1,
                                                          1,1,
                                                          0.05,
                                                          0.05,0.05,
                                                          0,0,0,0,]))
    
    # opt = minimize(objective_function, V, args=(x_c_l, y_c_l, z_c_l, x_c_r, y_c_r, z_c_r,
    #                                              cmd_lnt, cmd_lnp, cmd_unt, cmd_et, cmd_lep, cmd_rep,
    #                                              l_rvec, l_tvec, r_rvec, r_tvec), 
    #                                              method="Powell", 
    #                                              options={"disp":True},
    #                                              bounds=((-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(-1,1),
    #                                                      (0,4),(0,4),(0,4), (0,4),(0,4),(0,4),
    #                                                      (-1,1),
    #                                                      (0,1),(-1,0),(-1,0),(0,1),
    #                                                      (0,1),(0,1),
    #                                                      (0,0.05),
    #                                                      # (-0.05,0),
    #                                                      (0,0.05),(0,0.05),
    #                                                      (-3.1416,0),(-3.1416,0),(-3.1416,0),(-3.1416,0),)
    #                                             )
    
    
    print(dict(zip(var_names_list, opt.x)))

    # Saving URDF Results
    V = opt.x

    for joint in robot.joints:
        if joint.name == 'torso':
            joint.origin.position[0] = V[var2idx['torso_origin_x']]
            joint.origin.position[1] = V[var2idx['torso_origin_y']]
            joint.origin.position[2] = V[var2idx['torso_origin_z']]
        elif joint.name == 'neck_pitch':
            joint.origin.rotation[0] = V[var2idx['neck_pitch_rot_r']]
        elif joint.name == 'head_pitch':
            joint.origin.position[2] = V[var2idx['head_pitch_origin_z']]
            # joint.origin.rotation[0] = V[29]
            # joint.origin.rotation[2] = V[30]
        elif joint.name == 'eyes_pitch':
            joint.origin.position[0] = V[var2idx['eyes_pitch_origin_x']]
            joint.origin.position[2] = V[var2idx['eyes_pitch_origin_z']]
            # joint.origin.rotation[0] = V[33]
            # joint.origin.rotation[2] = V[34]
        elif joint.name == 'lefteye_yaw':
            joint.origin.position[1] = V[var2idx['lefteye_yaw_origin_y']]
        elif joint.name == 'righteye_yaw':
            joint.origin.position[1] = -V[var2idx['lefteye_yaw_origin_y']]
        elif joint.name == 'lefteye_cam':
            joint.origin.position[0] = V[var2idx['lefteye_cam_origin_x']]
            joint.origin.rotation[0] = V[var2idx['lefteye_cam_rot_r']]
            # joint.origin.rotation[1] = V[var2idx['lefteye_cam_rot_p']]
            joint.origin.rotation[2] = V[var2idx['lefteye_cam_rot_y']]
        elif joint.name == 'righteye_cam':
            joint.origin.position[0] = V[var2idx['righteye_cam_origin_x']]
            joint.origin.rotation[0] = V[var2idx['righteye_cam_rot_r']]
            # joint.origin.rotation[1] = V[var2idx['righteye_cam_rot_p']]
            joint.origin.rotation[2] = V[var2idx['righteye_cam_rot_y']]

    # Write the XML string to a .urdf file
    urdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..','urdf', '075m_grace_leastsquares_results.urdf')
    results_str = robot.to_xml_string()
    with open(urdf_path, 'w') as file:
        file.write(results_str)
    print(f"XML string saved to {urdf_path}")

    # Print and save the results to csv
    res_df = pd.DataFrame({
        'initial': var_list,
        'learned': V,
    })
    res_df.index = var_names_list
    print(res_df)
    res_fn = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    res_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..','results', res_fn+'_075m_grace_leastsquares_results.csv')
    res_df.to_csv(res_path)
    print(f"CSV results saved to {res_path}")


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print('Elapsed Time (sec):',end-start)