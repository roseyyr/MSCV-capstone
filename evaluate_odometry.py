import numpy as np
from get_pointclouds import construct_graph, get_clustered_point_cloud
import sys
sys.path.append("./tools")
from associate import associate, read_file_list
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import random
from scipy.spatial.transform import Rotation as R
import math
import odometry as odo

def convertAngle(angle) :
    angle = angle % (2*np.pi)
    for i in range(3) :
        if (angle[i] > np.pi) :
            angle[i] = angle[i] - 2*np.pi
    return angle

def quaternion_to_euler(quaternion_list):
    x,y,z,w = quaternion_list
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    X = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    Z = math.atan2(t3, t4)

    return np.array([X, Y, Z])

def euler_to_quaternion(eulerAngles):
    roll, pitch, yaw = eulerAngles
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return np.array([qx, qy, qz, qw])


# distance between orientations : https://math.stackexchange.com/questions/90081/quaternion-distance
def evaluate_odometry(rgbd_im1, rgbd_im2, gt1, gt2) :
    rgb_im1, depth_im1 = rgbd_im1

    gt1_float = [float(i) for i in gt1]
    gt1_trans = gt1_float[0:3]
    gt1_rot = gt1_float[3:]
    gt1_rot_euler = quaternion_to_euler(gt1_rot)

    gt2_float = [float(i) for i in gt2]
    gt2_trans = gt2_float[0:3]
    gt2_rot = gt2_float[3:]
    gt2_rot_euler = quaternion_to_euler(gt2_rot)

    ref_cloud_dict, ref_orb_dict = get_clustered_point_cloud(np.array(rgb_im1),np.array(depth_im1))
    graph = construct_graph(ref_cloud_dict)
    R_res, t_res = odo.calculate_odometry(rgb_im1, graph,ref_cloud_dict,ref_orb_dict,(rgbd_im2))

    orientation_meas = (gt1_rot_euler + R_res)
    orirentation_meas = convertAngle(orientation_meas)
    orientation_meas_quat = euler_to_quaternion(orientation_meas)

    trans_error = np.linalg.norm(gt1_trans + t_res - gt2_trans)
    rot_error = 1 - ((np.sum(gt2_rot * orientation_meas_quat))**2)
    #Should rot_error be abs ? Discuss and check.

    return rot_error, trans_error

if __name__ == "__main__":
    data_dir = "../rgbd_dataset_freiburg1_xyz/"

    depth_path = data_dir + "depth.txt"
    rgb_path = data_dir + "rgb.txt"
    gt_path = data_dir + "groundtruth.txt"

    depth_list = read_file_list(depth_path)
    rgb_list = read_file_list(rgb_path)
    gt_list = read_file_list(gt_path)

    asso_list = associate(rgb_list, depth_list, offset=0.0, max_difference=0.02)
    asso_list2 = associate(rgb_list, gt_list, offset=0.0, max_difference=0.02)

    list_len = min(len(asso_list), len(asso_list2))

    rot_consecutive = []
    trans_consecutive = []
    for index in range(list_len - 1) :
        print (index)
        rgb1 = data_dir + rgb_list[asso_list[index][0]][0]
        depth1 = data_dir + depth_list[asso_list[index][1]][0]
        gt1 = gt_list[asso_list2[0][1]]
    
        rgb_im1 = Image.open(rgb1)
        depth_im1 = Image.open(depth1)

        rgb2 = data_dir + rgb_list[asso_list[index+1][0]][0]
        depth2 = data_dir + depth_list[asso_list[index+1][1]][0]
        gt2 = gt_list[asso_list2[1][1]]

        rgb_im2 = Image.open(rgb2)
        depth_im2 = Image.open(depth2)
        R_error, t_error = evaluate_odometry((rgb_im1, depth_im1), (rgb_im2,depth_im2), gt1, gt2)
        rot_consecutive.append(R_error)
        trans_consecutive.append(t_error)

    '''
    rot_cum = []
    trans_cum = []
    R_error_cum = 0
    t_error_cum = 0
    for index in range(10) :
        rgb1 = data_dir + rgb_list[asso_list[0][0]][0]
        depth1 = data_dir + depth_list[asso_list[0][1]][0]
        gt1 = gt_list[asso_list2[0][1]]
    
        rgb_im1 = Image.open(rgb1)
        depth_im1 = Image.open(depth1)

        rgb2 = data_dir + rgb_list[asso_list[index][0]][0]
        depth2 = data_dir + depth_list[asso_list[index][1]][0]
        gt2 = gt_list[asso_list2[1][1]]

        rgb_im2 = Image.open(rgb2)
        depth_im2 = Image.open(depth2)
        R_error, t_error = evaluate_odometry((rgb_im1, depth_im1), (rgb_im2,depth_im2), gt1, gt2)
        R_error_cum += R_error
        t_error_cum += t_error 
        rot_cum.append(R_error_cum)
        trans_cum.append(t_error_cum)
    '''

    plt.plot(rot_consecutive, color = 'r', label='Rotation')
    plt.plot(trans_consecutive, color = 'g', label='Translation')
    plt.show()

