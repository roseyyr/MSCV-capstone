import numpy as np
from get_pointclouds import construct_graph, get_clustered_point_cloud
import sys
sys.path.append("./tools")
from associate import associate, read_file_list
from PIL import Image
import cv2
def calculate_odometry(ref_graph,ref_cloud_dict,ref_orb_dict,rgbd):
    reg_im, depth_im = rgbd
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
    for match_str,votes in sorted(match_dict.items(),key=lambda x:x[1]):
        arr = match_str.split("-")
        idx1, idx2 = int(arr[0]), int(arr[1])
        if votes>=10 and idx1 not in s1 and idx2 not in s2:
            res[idx1] = idx2
            s1.add(idx1)
            s2.add(idx2)
    print(res)


if __name__ == "__main__":
    data_dir = "/data/datasets/yurouy/rgbd_dataset_freiburg1_xyz/"
    depth_path = data_dir + "depth.txt"
    rgb_path = data_dir + "rgb.txt"
    depth_list = read_file_list(depth_path)
    rgb_list = read_file_list(rgb_path)
    asso_list = associate(rgb_list, depth_list,offset=0.0,max_difference=0.02)

    
    rgb = data_dir + rgb_list[asso_list[0][0]][0]
    depth = data_dir + depth_list[asso_list[0][1]][0]
    rgb_im = Image.open(rgb)
    depth_im = Image.open(depth)
    ref_cloud_dict, ref_orb_dict = get_clustered_point_cloud(np.array(rgb_im),np.array(depth_im))
    graph = construct_graph(ref_cloud_dict)

    rgb = data_dir + rgb_list[asso_list[1][0]][0]
    depth = data_dir + depth_list[asso_list[1][1]][0]
    rgb_im = Image.open(rgb)
    depth_im = Image.open(depth)
    calculate_odometry(graph,ref_cloud_dict,ref_orb_dict,(rgb_im,depth_im))


