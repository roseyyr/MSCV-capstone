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

def calc_pairwise_error(gt2, R_abs, t_abs) :
    gt2_float = [float(i) for i in gt2]
    gt2_trans = gt2_float[0:3]
    gt2_rot = gt2_float[3:]
    gt2_rot_euler = quaternion_to_euler(gt2_rot)
    print (t_abs, gt2_trans)
    trans_error = np.linalg.norm(t_abs - gt2_trans)
    rot_error = 1 - ((np.sum(gt2_rot * R_abs))**2)

    return rot_error, trans_error

def calc_Rot_and_Trans_world(gt1, R_rel, t_rel) :

    gt1_float = [float(i) for i in gt1]
    gt1_trans = gt1_float[0:3]
    gt1_rot = gt1_float[3:]
    gt1_rot_euler = quaternion_to_euler(gt1_rot)

    orientation_meas = (gt1_rot_euler + R_rel)
    orientation_meas = convertAngle(orientation_meas)
    orientation_meas_quat = euler_to_quaternion(orientation_meas)

    trans_abs = gt1_trans + t_rel

    return orientation_meas_quat,trans_abs

def calc_odometry_pairwise(rgbd_im1, rgbd_im2, step) :
    rgb_im1, depth_im1 = rgbd_im1

    ref_cloud_dict, ref_orb_dict = get_clustered_point_cloud(np.array(rgb_im1),np.array(depth_im1))
    graph = construct_graph(ref_cloud_dict)
    R_res, t_res = odo.calculate_odometry(rgb_im1, graph,ref_cloud_dict,ref_orb_dict,(rgbd_im2), step)

    return R_res, t_res

if __name__ == "__main__":


    data_dir = sys.argv[1]

    keyFrameStart = 0
    keyFrameEnd = 1000
    keyFrameInterval = 5

    measure_ate_and_rpe_error = True

    depth_path = data_dir + "depth.txt"
    rgb_path = data_dir + "rgb.txt"
    gt_path = data_dir + "groundtruth.txt"

    depth_list = read_file_list(depth_path)
    rgb_list = read_file_list(rgb_path)
    gt_list = read_file_list(gt_path)

    asso_list = associate(rgb_list, depth_list, offset=0.0, max_difference=0.02)
    asso_list2 = associate(rgb_list, gt_list, offset=0.0, max_difference=0.02)

    list_len = min(len(asso_list), len(asso_list2))

    if measure_ate_and_rpe_error == True :
        fileID_groundtruth = open(sys.argv[2], 'w')
        fileID_estimatedValue = open(sys.argv[3], 'w')
        for index in range(keyFrameStart, keyFrameEnd, keyFrameInterval) :
    
            rgb_keyframe = data_dir + rgb_list[asso_list[index][0]][0]
            depth_keyframe = data_dir + depth_list[asso_list[index][1]][0]
            gt_keyframe = (gt_list[asso_list2[index][1]])
            #print (gt_keyframe)
            rgb_im_keyframe = Image.open(rgb_keyframe)
            depth_im_keyframe = Image.open(depth_keyframe)
            rgbd_keyframe = (rgb_im_keyframe, depth_im_keyframe)
    
            for frame in range(index, index + keyFrameInterval) :
                gt_currentframe = (gt_list[asso_list2[frame][1]])
                gtList_currentframe = [str(asso_list2[frame][1])]
                gtList_currentframe.extend(gt_currentframe)
                for values in gtList_currentframe :
                    fileID_groundtruth.write(values + ' ')
                fileID_groundtruth.write('\n')
    
                rgb_currentframe = data_dir + rgb_list[asso_list[frame][0]][0]
                depth_currentframe = data_dir + depth_list[asso_list[frame][1]][0]
    
                rgb_im_currentframe = Image.open(rgb_currentframe)
                depth_im_currentframe = Image.open(depth_currentframe)
                rgbd_currentframe = (rgb_im_currentframe, depth_im_currentframe)
    
                R_rel, t_rel = calc_odometry_pairwise(rgbd_keyframe, rgbd_currentframe, frame)
                R_abs, t_abs = calc_Rot_and_Trans_world(gt_keyframe, R_rel, t_rel)
                #print (R_abs_quat, t_abs)
                estimateList_currentFrame = [str(asso_list2[frame][1])]
                estimateList_currentFrame.extend(t_abs)
                estimateList_currentFrame.extend(R_abs)
                for values in estimateList_currentFrame :
                    fileID_estimatedValue.write(str(values) + ' ')
                fileID_estimatedValue.write('\n')
    else :
        rot_consecutive = []
        trans_consecutive = []
        for index in range(keyFrameStart, keyFrameEnd) :
            rgb1 = data_dir + rgb_list[asso_list[index][0]][0]
            depth1 = data_dir + depth_list[asso_list[index][1]][0]
            gt1 = gt_list[asso_list2[index][1]]
    
            rgb_im1 = Image.open(rgb1)
            depth_im1 = Image.open(depth1)

            rgb2 = data_dir + rgb_list[asso_list[index+1][0]][0]
            depth2 = data_dir + depth_list[asso_list[index+1][1]][0]
            gt2 = gt_list[asso_list2[index+1][1]]

            rgb_im2 = Image.open(rgb2)
            depth_im2 = Image.open(depth2)
            R_rel, t_rel = calc_odometry_pairwise((rgb_im1, depth_im1), (rgb_im2,depth_im2), index)
            R_abs, t_abs = calc_Rot_and_Trans_world(gt1, R_rel, t_rel)

            R_error, t_error = calc_pairwise_error(gt2, R_abs, t_abs)
            #print(R_error, t_error)
            rot_consecutive.append(R_error)
            trans_consecutive.append(t_error)
        plt.plot(rot_consecutive, color = 'r', label='Rotation')
        plt.plot(trans_consecutive, color = 'g', label='Translation')
        plt.show()
