import os
import sys

import time
import json
import math
import numpy as np
import pandas as pd
import cv2

from scipy.spatial.transform import Rotation as R
from pytransform3d.urdf import UrdfTransformManager
from urdf_parser_py.urdf import URDF


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

def motor_cmd_joint(cmd_lnp,cmd_lnt,cmd_unt,cmd_et,cmd_lep,cmd_rep):  # degrees
    cmd_lnp = math.radians(cmd_lnp)
    cmd_lnt = math.radians(cmd_lnt)
    cmd_unt = math.radians(cmd_unt)
    cmd_et = math.radians(cmd_et)
    cmd_lep = math.radians(cmd_lep)
    cmd_rep = math.radians(cmd_rep)

    neck_pitch = cmd_lnt*0.5 + 0
    neck_yaw = cmd_lnp*0.5 + 0
    head_pitch = cmd_unt*0.5 + 0
    eyes_pitch = cmd_et*0.4 + 0
    lefteye_yaw = cmd_lep*1.6 + 0
    righteye_yaw = cmd_rep*1.7 + 0

    res_dict = {
        "neck_pitch": neck_pitch,
        "neck_yaw": neck_yaw,
        "head_pitch": head_pitch,
        "eyes_pitch": eyes_pitch,
        "lefteye_yaw": lefteye_yaw,
        "righteye_yaw": righteye_yaw,
    }

    return res_dict

def generate_chest_dataset(mat, z, x_prime, y_prime): # Opencv coordinates
    a = np.array([[mat[0,0],mat[0,1],-x_prime],
              [mat[1,0],mat[1,1],-y_prime],
              [mat[2,0],mat[2,1],-1]])
    b = np.array([-mat[0,2]*z-mat[0,3], -mat[1,2]*z-mat[1,3], -mat[2,2]*z-mat[2,3]])
    x = np.linalg.solve(a, b)
    return np.append(x[:2],z)

def mtx_to_rvectvec(T_input):
    rvec, _ = cv2.Rodrigues(T_input[:3,:3])
    tvec = T_input[:3,3]
    return rvec.flatten(), tvec.flatten()


def main():
    # Load Configs
    cam_mtxs = load_json('camera_mtx.json')
    gaze_ctrs = load_json('calib_params.json')
    urdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', "urdf","chest_grace_cam.urdf")
    robot = URDF.from_xml_file(urdf_path)

    # Robot Construction
    robot = URDF.from_xml_file(urdf_path)

    # URDF Variable Assignment
    for joint in robot.joints:
        if joint.name == 'torso':
            joint.origin.position[0] = 0.0325 + 0  # Offset
            joint.origin.position[1] = -0.05692 + 0
            joint.origin.position[2] = -0.12234 + 0
        elif joint.name == 'neck_pitch':
            joint.origin.rotation[0] = 0 + 0
        elif joint.name == 'head_pitch':
            joint.origin.position[2] = 0.13172 + 0
            joint.origin.rotation[0] = 0 + 0
            joint.origin.rotation[2] = 0 + 0
        elif joint.name == 'eyes_pitch':
            joint.origin.position[0] = 0.08492 + 0
            joint.origin.position[2] = 0.05186 + 0
            joint.origin.rotation[0] = 0 + 0
            joint.origin.rotation[2] = 0 + 0
        elif joint.name == 'lefteye_yaw':
            joint.origin.position[1] = 0.02895 + 0
        elif joint.name == 'righteye_yaw':
            joint.origin.position[1] = -0.02895 + 0
        elif joint.name == 'lefteye_cam':
            joint.origin.position[0] = 0.015 + 0
            joint.origin.rotation[0] = -1.5708 + 0.03457662014401592  # For Y, Downwards is + because of Camera Opencv Coordinate System
            joint.origin.rotation[1] = 0 + 0
            joint.origin.rotation[2] = -1.5708 + 0.18913797017088096  # Orientation Offset: Right Hand Rule (Positive CCW)
        elif joint.name == 'righteye_cam':
            joint.origin.position[0] = 0.015 + 0
            joint.origin.rotation[0] = -1.5708 + 0.05553511593216458  # For Y, Downwards is + because of Camera Opencv Coordinate System
            joint.origin.rotation[1] = 0 + 0
            joint.origin.rotation[2] = -1.5708 + -0.2544206094843146  # Orientation Offset: Right Hand Rule (Positive CCW)
            
    # XML to String
    urdf_str = xml_to_str(robot)

    # Transform Manager
    tm = UrdfTransformManager()
    tm.load_urdf(urdf_str)

    # Undistort Points
    left_gaze_pt_x = gaze_ctrs['left_eye']['x_center']
    left_gaze_pt_y = gaze_ctrs['left_eye']['y_center']
    right_gaze_pt_x = gaze_ctrs['right_eye']['x_center']
    right_gaze_pt_y = gaze_ctrs['right_eye']['y_center']

    # Normalized Point (z=1.0)
    left_undistort_pts = cv2.undistortPoints((left_gaze_pt_x, left_gaze_pt_y), 
                                            np.array(cam_mtxs['left_eye']['camera_matrix']), 
                                            np.array(cam_mtxs['left_eye']['distortion_coefficients']))
    right_undistort_pts = cv2.undistortPoints((right_gaze_pt_x, right_gaze_pt_y), 
                                            np.array(cam_mtxs['right_eye']['camera_matrix']), 
                                            np.array(cam_mtxs['right_eye']['distortion_coefficients']))
    
    # Undistored Points
    left_x_prime = left_undistort_pts.squeeze()[0]
    left_y_prime = left_undistort_pts.squeeze()[1]
    right_x_prime = right_undistort_pts.squeeze()[0]
    right_y_prime = right_undistort_pts.squeeze()[1]

    # Initialization
    z_depth = 1.0
    list_ep=list(range(-14,15,2))
    list_et=list(range(20,-31,-5))
    list_lnp=list(range(-35,36,5))
    list_lnt=list(range(-10,31,10))
    list_unt=list(range(40,-11,-10))
    data_dict = { 'x_c_l': [],
                'y_c_l': [],
                'z_c_l': [],
                'x_c_r': [],
                'y_c_r': [],
                'z_c_r': [],
                'cmd_theta_lower_neck_pan':[],
                'cmd_theta_lower_neck_tilt':[],
                'cmd_theta_upper_neck_tilt':[],
                'cmd_theta_left_eye_pan': [],
                'cmd_theta_right_eye_pan': [],
                'cmd_theta_eyes_tilt':[],
                'l_rvec_0': [],
                'l_rvec_1': [],
                'l_rvec_2': [],
                'l_tvec_0': [],
                'l_tvec_1': [],
                'l_tvec_2': [],
                'r_rvec_0': [],
                'r_rvec_1': [],
                'r_rvec_2': [],
                'r_tvec_0': [],
                'r_tvec_1': [],
                'r_tvec_2': [],
                'neck_pitch': [],
                'neck_yaw': [],
                'head_pitch': [],
                'eyes_pitch': [],
                'lefteye_yaw': [],
                'righteye_yaw': [],
                }

    # Data Generation
    for lnt in list_lnt:
        for unt in list_unt:
            for lnp in list_lnp:
                for et in list_et:
                    for ep in list_ep:
                        # Robot Commands
                        joint_dict = motor_cmd_joint(cmd_lnp=lnp,cmd_lnt=lnt,cmd_unt=unt,cmd_et=et,cmd_lep=ep,cmd_rep=ep)
                        neck_pitch = joint_dict['neck_pitch']
                        neck_yaw = joint_dict['neck_yaw']
                        head_pitch = joint_dict['head_pitch']
                        eyes_pitch = joint_dict['eyes_pitch']
                        lefteye_yaw = joint_dict['lefteye_yaw']
                        righteye_yaw = joint_dict['righteye_yaw']

                        # Setting Robot Joints
                        tm.set_joint('neck_pitch', neck_pitch)
                        tm.set_joint('neck_yaw', neck_yaw)
                        tm.set_joint('head_pitch', head_pitch)
                        tm.set_joint('eyes_pitch', eyes_pitch)
                        tm.set_joint('lefteye_yaw', lefteye_yaw)
                        tm.set_joint('righteye_yaw', righteye_yaw)
                        
                        # Get Transform
                        T_clprime = tm.get_transform('realsense','leftcamera')
                        T_crprime = tm.get_transform('realsense','rightcamera')
                    
                        # Get Camera Chest Points
                        X_C_L = generate_chest_dataset(mat=T_clprime, z=z_depth, x_prime=left_x_prime, y_prime=left_y_prime)
                        X_C_R = generate_chest_dataset(mat=T_crprime, z=z_depth, x_prime=right_x_prime, y_prime=right_y_prime)

                        # Get quat and tvec
                        l_rvec, l_tvec = mtx_to_rvectvec(T_clprime)
                        r_rvec, r_tvec = mtx_to_rvectvec(T_crprime)

                        # Saving Data
                        data_dict['x_c_l'].append(X_C_L[0])
                        data_dict['y_c_l'].append(X_C_L[1])
                        data_dict['z_c_l'].append(X_C_L[2])
                        data_dict['x_c_r'].append(X_C_R[0])
                        data_dict['y_c_r'].append(X_C_R[1])
                        data_dict['z_c_r'].append(X_C_R[2])
                        data_dict['cmd_theta_lower_neck_tilt'].append(lnt)
                        data_dict['cmd_theta_lower_neck_pan'].append(lnp)
                        data_dict['cmd_theta_upper_neck_tilt'].append(unt)
                        data_dict['cmd_theta_left_eye_pan'].append(ep)
                        data_dict['cmd_theta_right_eye_pan'].append(ep)
                        data_dict['cmd_theta_eyes_tilt'].append(et)

                        data_dict['l_rvec_0'].append(l_rvec[0])
                        data_dict['l_rvec_1'].append(l_rvec[1])
                        data_dict['l_rvec_2'].append(l_rvec[2])
                        data_dict['l_tvec_0'].append(l_tvec[0])
                        data_dict['l_tvec_1'].append(l_tvec[1])
                        data_dict['l_tvec_2'].append(l_tvec[2])
                        data_dict['r_rvec_0'].append(r_rvec[0])
                        data_dict['r_rvec_1'].append(r_rvec[1])
                        data_dict['r_rvec_2'].append(r_rvec[2])
                        data_dict['r_tvec_0'].append(r_tvec[0])
                        data_dict['r_tvec_1'].append(r_tvec[1])
                        data_dict['r_tvec_2'].append(r_tvec[2])

                        data_dict['neck_pitch'].append(neck_pitch)
                        data_dict['neck_yaw'].append(neck_yaw)
                        data_dict['head_pitch'].append(head_pitch)
                        data_dict['eyes_pitch'].append(eyes_pitch)
                        data_dict['lefteye_yaw'].append(lefteye_yaw)
                        data_dict['righteye_yaw'].append(righteye_yaw)

    # Pandas Dataframe
    data_df = pd.DataFrame(data_dict)
    print(len(data_df))

    # Saving CSV
    
    sim_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..','data','sim','240924_sim_dataset.csv')
    data_df.to_csv(sim_data_path , index=False)
    print('Saved to:',sim_data_path)


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print('Elapsed Time (sec):',end-start)
    