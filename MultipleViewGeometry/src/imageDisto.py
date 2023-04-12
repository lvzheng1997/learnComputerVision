import cv2
import numpy as np

file_name = "H:\\public\\001816_84776315576560.jpg"

img2 = cv2.imread(file_name,0)
h, w = img2.shape[:2]
print(h, w )

fx = 948.8
fy = 948.8
cx = 1241.49
cy = 1037.85

mtx = np.zeros((3,3))
mtx[0,0] = 2399.943867776368
mtx[1,1] = 2399.5474015190184
mtx[0,2] = 1021.6738324110975
mtx[1,2] = 758.3128102787988
mtx[2,2] = 1
print(mtx)

dist = np.array([[-0.20950630193675363, 0.1564084886038431, -4.228041576380835e-06, 0.00014212455311098375]])
                 # -0.719893 -0.0363548 0.183786 948.8 1241.49 1038.5 458.974 1242.15 1037.85
#k[0],          k[1],          k[2],         k[3],      p[0],          p[1],    c[0],      c[1],    lamda,FocalLengthInitial,   Ip0[0] ,Ip0[1],   frectified,x0rectified,y0rectified;
#9.91677e-07 -4.77368e-13 7.29719e-20 -2.51502e-27 -8.11214e-05 -0.000105306 -0.719893 -0.0363548  0.183786        948.8        1241.49  1038.5    458.974      1242.15     1037.85
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
#dst=cv2.omnidir.undistortImage(img2, mtx,  dist,3.53696702641, cv2.RECTIFY_PERSPECTIVE, np.zeros((3,3)), (w, h))
dst = cv2.undistort(img2, mtx, dist, None, newcameramtx)
print(dst.shape)

cv2.imshow("dst",dst)
cv2.waitKey()