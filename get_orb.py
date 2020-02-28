import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
def extract_orb(image):
    kp = []
    des = []
    H, W, _ = image.shape
    # H4, W4 = H//4, W//4
    # for i in range(4):
        # for j in range(4):
            # orb = cv2.ORB_create(nfeatures=2000)
            # im = image[i*H4:(i+1)*H4,j*W4:(j+1)*W4]
            # tmpkp, tmpdes = orb.detectAndCompute(im,None)
            # for k in tmpkp:
                # point = (k.pt[0]+j*W4, k.pt[1]+i*H4)
                # k.pt = point
            # if tmpdes is not None:
                # kp += tmpkp
                # des.append(tmpdes)
    
    # des = np.vstack(tuple(des)).reshape((-1,32))
    orb = cv2.ORB_create(nfeatures=10000)
    kp, des = orb.detectAndCompute(image,None)
    kp, des = np.array(kp), np.array(des)
    # print(des.shape[0])
    # img2 = cv2.drawKeypoints(image,kp,outImage=None,color=(0,255,0), flags=0)
    # plt.imshow(img2)
    # plt.savefig("orb_result")
    return kp, des

if __name__=="__main__":
    image = cv2.imread("/data/datasets/yurouy/rgbd_dataset_freiburg1_xyz/rgb/1305031102.175304.png")
    kp, des = extract_orb(image)
    image2 = cv2.drawKeypoints(image,kp,outImage=None,color=(255,0,0),flags=0)
