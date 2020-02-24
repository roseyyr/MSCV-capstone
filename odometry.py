import numpy as np
from get_pointclouds import construct_graph, get_clustered_point_cloud
import sys
sys.path.append("./tools")
from associate import associate, read_file_list
from PIL import Image
def calculate_odometry(ref_graph,ref_cloud_dict,rgbd):
    reg_im, depth_im = rgbd
    cloud_dict = get_clustered_point_cloud(np.array(rgb_im),np.array(depth_im))
    graph = construct_graph(cloud_dict)
    print(ref_graph)
    print(graph)


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
    ref_cloud_dict = get_clustered_point_cloud(np.array(rgb_im),np.array(depth_im))
    graph = construct_graph(ref_cloud_dict)

    rgb = data_dir + rgb_list[asso_list[1][0]][0]
    depth = data_dir + depth_list[asso_list[1][1]][0]
    rgb_im = Image.open(rgb)
    depth_im = Image.open(depth)
    calculate_odometry(graph,ref_cloud_dict,(rgb_im,depth_im))


