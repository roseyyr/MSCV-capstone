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
    im = cv2.imread("./input.jpg")
    res = get_segres(im)
