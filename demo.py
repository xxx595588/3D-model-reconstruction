import numpy as np
import matplotlib.pyplot as plt
import pickle
import cv2
from camutils import *
import meshutils

# load in the intrinsic camera parameters from 'calibration.pickle'
file = open("calibration.pickle", "rb")
data = pickle.load(file)
f = (data["fx"]+data["fy"])/2
c = np.array([[data["cx"]], [data["cy"]]])
t=np.array([[0, 0, 0]])
R=makerotation(0, 0, 0)

# create Camera objects representing the left and right cameras
# use the known intrinsic parameters you loaded in.
camL = Camera(f, c, R, t)
camR = Camera(f, c, R, t)

# load in the left and right images and find the coordinates of
# the chessboard corners using OpenCV
imgL = plt.imread('calib_jpg_u/frame_C0_02.jpg')
ret, cornersL = cv2.findChessboardCorners(imgL, (8,6), None)
pts2L = cornersL.squeeze().T

imgR = plt.imread('calib_jpg_u/frame_C1_02.jpg')
ret, cornersR = cv2.findChessboardCorners(imgR, (8,6), None)
pts2R = cornersR.squeeze().T

# generate the known 3D point coordinates of points on the checkerboard in cm
pts3 = np.zeros((3,6*8))
yy,xx = np.meshgrid(np.arange(8),np.arange(6))
pts3[0,:] = 2.8*xx.reshape(1,-1)
pts3[1,:] = 2.8*yy.reshape(1,-1)


# Now use your calibratePose function to get the extrinsic parameters
# for the two images. You may need to experiment with the initialization
# in order to get a good result
init_L = np.array([0,1,0,0,0,-2]) 
init_R = np.array([0,1,0,0,0,-2]) 

camL = calibratePose(pts3,pts2L,camL,init_L)
camR = calibratePose(pts3,pts2R,camR,init_R)

# As a final test, triangulate the corners of the checkerboard to get back there 3D locations
pts3r = triangulate(pts2L,camL,pts2R,camR)

threshold = 0.01
threshold_obj = 0.005
step = 3
trithresh = 0.5
boxlimit = np.array([-1,18,1,25,15,30])
obj_name = "teapot"

print("Processing grab0")
imprefixL_obj, imprefixR_obj = obj_name+'/grab_0_u/color_C0_', obj_name+'/grab_0_u/color_C1_'
imprefixL, imprefixR = obj_name+'/grab_0_u/frame_C0_', obj_name+'/grab_0_u/frame_C1_' 
pts2L,pts2R,pts3,color_value = reconstruct(imprefixL,imprefixR,threshold,imprefixL_obj,imprefixR_obj,threshold_obj,camL,camR)
pts3,tri,color_value=mesh(pts2L,pts2R,pts3,boxlimit,step,color_value,trithresh)
meshutils.writeply(pts3,color_value,tri,'grab0.ply')


print("Processing grab1")
imprefixL_obj, imprefixR_obj = obj_name+'/grab_1_u/color_C0_', obj_name+'/grab_1_u/color_C1_'
imprefixL, imprefixR = obj_name+'/grab_1_u/frame_C0_', obj_name+'/grab_1_u/frame_C1_' 
pts2L,pts2R,pts3,color_value = reconstruct(imprefixL,imprefixR,threshold,imprefixL_obj,imprefixR_obj,threshold_obj,camL,camR)
pts3,tri,bvalues=mesh(pts2L,pts2R,pts3,boxlimit,step,color_value,trithresh)
meshutils.writeply(pts3,bvalues,tri,'grab1.ply')

print("Processing grab2")
imprefixL_obj, imprefixR_obj = obj_name+'/grab_2_u/color_C0_', obj_name+'/grab_2_u/color_C1_'
imprefixL, imprefixR = obj_name+'/grab_2_u/frame_C0_', obj_name+'/grab_2_u/frame_C1_' 
pts2L,pts2R,pts3,color_value = reconstruct(imprefixL,imprefixR,threshold,imprefixL_obj,imprefixR_obj,threshold_obj,camL,camR)
pts3,tri,bvalues=mesh(pts2L,pts2R,pts3,boxlimit,step,color_value,trithresh)
meshutils.writeply(pts3,bvalues,tri,'grab2.ply')


print("Processing grab3")
imprefixL_obj, imprefixR_obj = obj_name+'/grab_3_u/color_C0_', obj_name+'/grab_3_u/color_C1_'
imprefixL,imprefixR  = obj_name+'/grab_3_u/frame_C0_', obj_name+'/grab_3_u/frame_C1_' 
pts2L,pts2R,pts3,color_value = reconstruct(imprefixL,imprefixR,threshold,imprefixL_obj,imprefixR_obj,threshold_obj,camL,camR)
pts3,tri,bvalues=mesh(pts2L,pts2R,pts3,boxlimit,step,color_value,trithresh)
meshutils.writeply(pts3,bvalues,tri,'grab3.ply')


print("Processing grab4")
imprefixL_obj, imprefixR_obj = obj_name+'/grab_4_u/color_C0_', obj_name+'/grab_4_u/color_C1_'
imprefixL, imprefixR = obj_name+'/grab_4_u/frame_C0_', obj_name+'/grab_4_u/frame_C1_'
pts2L,pts2R,pts3,color_value = reconstruct(imprefixL,imprefixR,threshold,imprefixL_obj,imprefixR_obj,threshold_obj,camL,camR)
pts3,tri,bvalues=mesh(pts2L,pts2R,pts3,boxlimit,step,color_value,trithresh)
meshutils.writeply(pts3,bvalues,tri,'grab4.ply')


print("Processing grab5")
imprefixL_obj, imprefixR_obj = obj_name+'/grab_5_u/color_C0_', obj_name+'/grab_5_u/color_C1_'
imprefixL, imprefixR = obj_name+'/grab_5_u/frame_C0_', obj_name+'/grab_5_u/frame_C1_' 
pts2L,pts2R,pts3,color_value = reconstruct(imprefixL,imprefixR,threshold,imprefixL_obj,imprefixR_obj,threshold_obj,camL,camR)
pts3,tri,bvalues=mesh(pts2L,pts2R,pts3,boxlimit,step,color_value,trithresh)
meshutils.writeply(pts3,bvalues,tri,'grab5.ply')

print("Processing grab6")
imprefixL_obj, imprefixR_obj = obj_name+'/grab_6_u/color_C0_', obj_name+'/grab_6_u/color_C1_'
imprefixL, imprefixR = obj_name+'/grab_6_u/frame_C0_', obj_name+'/grab_6_u/frame_C1_' 
pts2L,pts2R,pts3,color_value = reconstruct(imprefixL,imprefixR,threshold,imprefixL_obj,imprefixR_obj,threshold_obj,camL,camR)
pts3,tri,bvalues=mesh(pts2L,pts2R,pts3,boxlimit,step,color_value,trithresh)
meshutils.writeply(pts3,bvalues,tri,'grab6.ply')