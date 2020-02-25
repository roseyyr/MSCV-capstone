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
    pred_boxes, pred_classes, scores, pred_masks = get_segres(np.array(rgb_im))
    kp, des = extract_orb(rgb_im)
    # img2 = cv2.drawKeypoints(rgb_im,kp,outImage=None,color=(0,255,0), flags=0)
    # for i in range(pred_boxes.tensor.size()[0]):
        # x1, y1, x2, y2 = pred_boxes.tensor[i]
        # x1, y1, x2, y2 = int(x1.item()), int(y1.item()), int(x2.item()), int(y2.item())
        # img2 = cv2.rectangle(img2,(x1,y1),(x2,y2),color=(255,0,0),thickness=2)
    # plt.imshow(img2)
    # plt.savefig("overlap.jpg")
    print("number of keypoints:",len(kp))    
    print("scores:",scores) 
    
    print("number of object:",pred_boxes.shape[0])
    cloud_dict = {}
    for i in range(pred_boxes.shape[0]):
        mask = pred_masks[i]
        cluster_dict = {"score":scores[i].item(), "pred_class":pred_classes[i].item(),"pred_box":pred_boxes[i],\
                        "color_arr":[], "xyz_arr":[],"des_arr":[]}
        cloud_dict[i] = cluster_dict

    assign = [-1 for i in range(len(kp))]
    for i in range(len(kp)):
        x, y = int(kp[i].pt[0]), int(kp[i].pt[1])
        label = None
        for j in range(len(pred_masks)):
            if pred_masks[j,y,x] == True:
                label = j
                break
 
        if label == None:
            continue
        color = rgb_im[y,x]
        Z = depth_im[y,x] / scalingFactor
        if Z==0: 
            continue
        X = (x - centerX) * Z / focalLength
        Y = (y - centerY) * Z / focalLength
       
        cloud_dict[label]["color_arr"].append([color[0],color[1],color[2]])
        cloud_dict[label]["xyz_arr"].append([X,Y,Z])
        cloud_dict[label]["des_arr"].append(des[i])
        assign[i] = label

    delete_arr = []   
    for i in cloud_dict:
        orb_num = len(cloud_dict[i]["color_arr"])
        print(i,orb_num)
        if orb_num < 10:
            delete_arr.append(i)
    print("delete:",delete_arr)
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
        depth = data_dir + depth_list[asso_list[i][1]][0]
        rgb_im = Image.open(rgb)
        depth_im = Image.open(depth)
        get_clustered_point_cloud(np.array(rgb_im),np.array(depth_im))
