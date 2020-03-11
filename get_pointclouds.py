import sys
sys.path.append("./tools")
import numpy as np
from associate import associate, read_file_list
from PIL import Image
from get_segres import get_segres
from get_orb import extract_orb
import cv2
import matplotlib.pyplot as plt
scalingFactor = 5000.0
focalLength = 525.0
centerX = 319.5
centerY = 239.5
min_orb = 10


def construct_graph(cloud_dict):
    
    graph = {}
    centers = {}
    for idx in cloud_dict:
        cluster = cloud_dict[idx]
        centroid = np.mean(cluster["xyz_arr"],axis=0)
        centers[idx] = centroid
    
    for i in cloud_dict:
        graph[i] = {}
        for j in cloud_dict:
            if i!=j:
                graph[i][j] = np.sum((centers[i]-centers[j])**2)
    return graph
 

def get_clustered_point_cloud(rgb_im,depth_im):
    H, W, _ = rgb_im.shape
    panoptic_seg, seg_info = get_segres(np.array(rgb_im))
    kp, des = extract_orb(rgb_im)
    # img2 = cv2.drawKeypoints(rgb_im,kp,outImage=None,color=(0,255,0), flags=0)
    # for i in range(pred_boxes.tensor.size()[0]):
        # x1, y1, x2, y2 = pred_boxes.tensor[i]
        # x1, y1, x2, y2 = int(x1.item()), int(y1.item()), int(x2.item()), int(y2.item())
        # img2 = cv2.rectangle(img2,(x1,y1),(x2,y2),color=(255,0,0),thickness=2)
    # plt.imshow(img2)
    # plt.savefig("overlap.jpg")
    #print("number of keypoints:",len(kp))    
    #print("scores:",scores) 
    
    #print("number of object:",pred_boxes.shape[0])
    cloud_dict = {}
    N = len(seg_info)
    # print(seg_info)
    # print(panoptic_seg)
    for i in range(N):
        cluster_dict = {"pred_class":seg_info[i]['category_id'], \
                        "color_arr":[], "xyz_arr":[],"des_arr":[],"uv_arr":[],"kp_arr":[]}
        if seg_info[i]['isthing']:
            cluster_dict['score'] = seg_info[i]['score']
        cloud_dict[i+1] = cluster_dict

    assign = [-1 for i in range(len(kp))]
    for i in range(len(kp)):
        v, u = int(kp[i].pt[0]), int(kp[i].pt[1])
        label = panoptic_seg[u, v]
        if 1 <= label <= N:
            color = rgb_im[u,v]
            Z = depth_im[u,v] / scalingFactor
            if Z==0: 
                continue
            X = (u - centerX) * Z / focalLength
            Y = (v - centerY) * Z / focalLength
       
            cloud_dict[label]["color_arr"].append([color[0],color[1],color[2]])
            cloud_dict[label]["xyz_arr"].append([X,Y,Z])
            cloud_dict[label]["des_arr"].append(des[i])
            cloud_dict[label]["uv_arr"].append([u,v])
            cloud_dict[label]["kp_arr"].append(kp[i])
            assign[i] = label

    delete_arr = []   
    for i in cloud_dict:
        orb_num = len(cloud_dict[i]["color_arr"])
        #print(i,orb_num)
        if orb_num < min_orb:
            delete_arr.append(i)
    #print("delete:",delete_arr)
    for i in delete_arr:
        del cloud_dict[i]

    orb_dict = {"kp":kp, "des":des, "assign":assign}
    
    return cloud_dict, orb_dict
    


if __name__ == "__main__":
    data_dir = "/data/datasets/yurouy/rgbd_dataset_freiburg1_xyz/"
    depth_path = data_dir + "depth.txt"
    rgb_path = data_dir + "rgb.txt"    
    depth_list = read_file_list(depth_path)
    rgb_list = read_file_list(rgb_path)
    asso_list = associate(rgb_list, depth_list,offset=0.0,max_difference=0.02)
  
    for i in range(1):
        rgb = data_dir + rgb_list[asso_list[i][0]][0]
        print(rgb)
        depth = data_dir + depth_list[asso_list[i][1]][0]
        rgb_im = Image.open(rgb)
        depth_im = Image.open(depth)
        get_clustered_point_cloud(np.array(rgb_im),np.array(depth_im))
