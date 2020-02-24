import cv2
import numpy as np
from matplotlib import pyplot as plt

def extract_orb(image):
    orb = cv2.ORB_create(nfeatures=2000)
    kp = orb.detect(image,None)
    kp, des = orb.compute(image,kp)
    # img2 = cv2.drawKeypoints(image,kp,outImage=None,color=(0,255,0), flags=0)
    # plt.imshow(img2)
    # plt.savefig("orb_result")
    return kp, des

if __name__=="__main__":
    image = cv2.imread('input.jpg',0)
    kp, des = extract_orb(image)
    image2 = cv2.drawKeypoints(image,kp,outImage=None,color=(255,0,0),flags=0)
