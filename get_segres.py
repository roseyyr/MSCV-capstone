import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import matplotlib.pyplot as plt
from os import listdir
cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
predictor = DefaultPredictor(cfg)
cnt = 0

def get_segres(img):
    global cnt
    outputs = predictor(img)
    panoptic_seg, seg_info = outputs['panoptic_seg']
    # print('number of instances detected:',len(seg_info))
    # v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    # v = v.draw_instance_predictions(outputs["instances"].to("cpu"))   
    # v = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), seg_info)
    # plt.imshow(v.get_image()[:,:,::-1])
    # plt.savefig("seg_res"+str(cnt)+".jpg")
    # cnt += 1
    return panoptic_seg.cpu().numpy(), seg_info

if __name__ == "__main__":
    input_dir = "/data/datasets/yurouy/rgbd_dataset_freiburg3_walking_xyz/rgb/"
    output_dir = "/data/datasets/yurouy/rgbd_dataset_freiburg3_walking_xyz/segment/"

    im_list = listdir(input_dir)
    H = 480
    W = 640
    for i in range(len(im_list)):
        print(i)
        im = cv2.imread(input_dir+im_list[i])
        res, seg_info = get_segres(im)
        output_path = output_dir + im_list[i][:-4] + ".msk"
        with open(output_path, "w") as f:
            for i in range(H):
                f.write(" ".join(map(str, res[i])) + "\n")
            f.write("instances:%d" % len(seg_info) + "\n")
            for i in range(len(seg_info)):
                seg_id = seg_info[i]['id']
                isthing = seg_info[i]['isthing']
                category = seg_info[i]['category_id']
                if isthing:
                    score = seg_info[i]['score']
                    arr = [seg_id, isthing, category, score]
                else:
                    arr = [seg_id, isthing, category]
                f.write(" ".join(map(str, arr)) + "\n")
        
