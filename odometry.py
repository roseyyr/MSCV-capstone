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
min_votes = 10
max_iter = 100
trans_threshold = 0.01
rot_threshold = 0.01

def rotation_ransac(l):
    res = None
    max_per = 0.0
    for i in range(max_iter):
        element = l[random.randint(0,len(l)-1)]
        new_l = l - element
        inliers = np.sum(new_l**2, axis=1).reshape(-1,) < rot_threshold
        if len(inliers)/len(l) > max_per:
            idx = np.nonzero(inliers)[0]
            element = np.mean(l[idx], axis=0)
            res = element
            new_l = l-element
            inliers = np.sum(new_l**2, axis=1) < rot_threshold
            max_per = len(inliers)/len(l)
    print('r_ransac:',res,'max percentage:',max_per)
    return res


def trans_ransac(l):
    res = None
    max_per = 0.0
    for i in range(max_iter):
        element = l[random.randint(0,len(l)-1)]
        new_l = l - element
        inliers = np.sum(new_l**2, axis=1).reshape(-1,) < trans_threshold
        if len(inliers)/len(l) > max_per:
            idx = np.nonzero(inliers)[0]
            element = np.mean(l[idx], axis=0)
            res = element 
            new_l = l - element
            inliers = np.sum(new_l**2, axis=1) < trans_threshold
            max_per = len(inliers)/len(l)
    print('t_ransac res:',res,'max percentage:', max_per)
    return res
              
        

def calculate_odometry(im1,im2,ref_graph,ref_cloud_dict,ref_orb_dict,rgbd):
    rgb_im, depth_im = rgbd
    cloud_dict,orb_dict = get_clustered_point_cloud(np.array(rgb_im),np.array(depth_im))
    graph = construct_graph(cloud_dict)
    
    # This section is for converting ORB Match --> Object Match
    match_dict = {}
    kp1, des1, assign1 = ref_orb_dict["kp"], ref_orb_dict["des"], ref_orb_dict['assign']
    kp2, des2, assign2 = orb_dict["kp"], orb_dict["des"],orb_dict['assign']
   
    # BFMatcher with default params
    bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)
    
    for i,m in enumerate(matches):
        idx1, idx2 = m.queryIdx, m.trainIdx
        l1, l2 = assign1[idx1], assign2[idx2]
        if l1==-1 or l2==-1:
            continue
        match_str = str(l1)+'-'+str(l2)
        if match_str not in match_dict:
            match_dict[match_str] = 1
        else:
            match_dict[match_str] += 1
    print(match_dict)
    res = {}
    s1, s2 = set(),set()

    for match_str,votes in sorted(match_dict.items(),key=lambda x:-x[1]):
        arr = match_str.split("-")
        idx1, idx2 = int(arr[0]), int(arr[1])
        if votes >= min_votes and idx1 not in s1 and idx2 not in s2:
            res[idx1] = idx2
            s1.add(idx1)
            s2.add(idx2)
    print(res)

    Rs, Ts = [], []
    K = cv2.UMat(np.array([[525.0,0.0,319.5],[0.0,525.0,239.5],[0.0,0.0,1.0]],dtype=np.float32))
    for o_idx1, o_idx2 in res.items():
        print(o_idx1, o_idx2)
        des1, des2 = ref_cloud_dict[o_idx1]['des_arr'], cloud_dict[o_idx2]['des_arr']
        des1, des2 = cv2.UMat(np.array(des1,dtype=np.uint8)), cv2.UMat(np.array(des2,dtype=np.uint8))
        tmp_matches = bf.match(des1, des2)
        tmp_matches = sorted(tmp_matches, key=lambda x:x.distance)
        num = len(tmp_matches)
        # im3 = cv2.drawMatches(cv2.UMat(np.array(im1)),kp1,cv2.UMat(np.array(im2)),kp2,tmp_matches[:num],outImg=None)  
        # plt.imshow(im3.get())
        # plt.savefig(str(o_idx1)+"-"+str(o_idx2)+".jpg")
        # plt.show()
        xyz_arr, uv_arr = [], []
        K = np.array([[525.0,0.0,319.5],[0.0,525.0,239.5],[0.0,0.0,1.0]])
        for i in range(num):
            m = tmp_matches[i]
            f_idx1, f_idx2 = m.queryIdx, m.trainIdx
            xyz_arr.append(ref_cloud_dict[o_idx1]['xyz_arr'][f_idx1])
            uv_arr.append(np.array(cloud_dict[o_idx2]['uv_arr'][f_idx2]))
        print('number of matches:', np.array(xyz_arr).shape[0])
        xyz_arr, uv_arr = cv2.UMat(np.array(xyz_arr,dtype=np.float32)), cv2.UMat(np.array(uv_arr,dtype=np.float32))
        rvec, tvec = cv2.UMat(np.array([0.0,0.0,0.0])), cv2.UMat(np.array([0.0,0.0,0.0]))
        flag, rvec, tvec, inliers = cv2.solvePnPRansac(xyz_arr,uv_arr,K,None,rvec,tvec,True)
        print('rotation:',rvec.get())
        print('translation:',tvec.get())
        print("number of inliers:",len(inliers.get()))
        Rs.append(rvec.get())
        Ts.append(tvec.get())
    RMs = []
    for rvec in Rs:
        r = R.from_rotvec(rvec.reshape(3,))
        RMs.append(r.as_euler('xyz'))
    R_res = rotation_ransac(np.array(RMs))
    t_res = trans_ransac(np.array(Ts))


if __name__ == "__main__":
    data_dir = "/data/datasets/yurouy/rgbd_dataset_freiburg1_xyz/"
    depth_path = data_dir + "depth.txt"
    rgb_path = data_dir + "rgb.txt"
    depth_list = read_file_list(depth_path)
    rgb_list = read_file_list(rgb_path)
    asso_list = associate(rgb_list, depth_list,offset=0.0,max_difference=0.02)

    
    rgb1 = data_dir + rgb_list[asso_list[0][0]][0]
    depth1 = data_dir + depth_list[asso_list[0][1]][0]
    rgb_im1 = Image.open(rgb1)
    depth_im1 = Image.open(depth1)
    ref_cloud_dict, ref_orb_dict = get_clustered_point_cloud(np.array(rgb_im1),np.array(depth_im1))
    graph = construct_graph(ref_cloud_dict)

    rgb2 = data_dir + rgb_list[asso_list[1][0]][0]
    depth2 = data_dir + depth_list[asso_list[1][1]][0]
    rgb_im2 = Image.open(rgb2)
    depth_im2 = Image.open(depth2)
    calculate_odometry(rgb_im1, rgb_im2, graph,ref_cloud_dict,ref_orb_dict,(rgb_im2,depth_im2))
    print('ground truth:')


