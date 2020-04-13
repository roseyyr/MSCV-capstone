import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
import math


num_quaternions = 5
use_random = False

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

if use_random == True :
	Q = np.zeros((4, num_quaternions))
	for i in range(num_quaternions) :
		u,v,w = np.random.random((3,1))
		quat = np.array([np.sqrt(1-u)*np.sin(2*np.pi*v), np.sqrt(1-u)*np.cos(2*np.pi*v), np.sqrt(u)*np.sin(2*np.pi*w), np.sqrt(u)*np.cos(2*np.pi*w)])
		Q[:,i] = np.squeeze(quat)
	Q = 1/(num_quaternions)*Q
	Qsquare = np.matmul(Q, np.transpose(Q))
	val, vec = np.linalg.eig(Qsquare)
	maxIndex = np.argmax(val)
	avgVec = vec[:,maxIndex]
	print (avgVec)
else :
	#Enter the Euler angles to be averaged here
	eulerAngles = np.array([[45,45,60],
	[100,150, 175],
	[60,55,67]])
	eulerAngles_rescaled = eulerAngles/360
	num_quaternions = eulerAngles.shape[1]
	Q = np.zeros((4, num_quaternions))
	for i in range(num_quaternions) :
		quat = euler_to_quaternion(convertAngle(eulerAngles_rescaled[i,:]))
		Q[:,i] = np.squeeze(quat)
	Q = 1/(num_quaternions)*Q
	Qsquare = np.matmul(Q, np.transpose(Q))
	val, vec = np.linalg.eig(Qsquare)
	maxIndex = np.argmax(val)
	avgVec = vec[:,maxIndex]
	eulerAngle = convertAngle(quaternion_to_euler(avgVec))*360
	print (eulerAngle)