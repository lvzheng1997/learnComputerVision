import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
def cat2images(limg, rimg):
    HEIGHT = limg.shape[0]
    WIDTH = limg.shape[1]
    imgcat = np.zeros((HEIGHT, WIDTH*2+20,3))
    imgcat[:,:WIDTH,:] = limg
    imgcat[:,-WIDTH:,:] = rimg
    for i in range(int(HEIGHT / 32)):
        imgcat[i*32,:,:] = 255
    return imgcat


images = glob.glob('E:\\Dateset\\CameraCalibrationDataset-main\\CameraCalibrationDataset-main\\Cannon 40D Lens Cannon 28-90mm\\*.jpg')#读取图像文件
print(images)
#images = images[0:2]
w, h = 9,6
i = 0

# 角点精确检测阈值
cv2.threshold(dst,1,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 找到棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
    print(ret)
    if ret == True:
        # 角点精确检测
        i += 1
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        # 将角点在图像上显示
        cv2.drawChessboardCorners(img, (w, h), corners, ret)

# 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵
objp = np.zeros((w * h, 3), np.float32)
objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)

# 储存棋盘格角点的世界坐标和图像坐标对
objpoints = []  # 在世界坐标系中的三维点
imgpoints = []  # 在图像平面的二维点
for fname in images:

    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 找到棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
    # 如果找到足够点对，将其存储起来
    if ret == True:
        # 角点精确检测
        i += 1
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners)

gray = cv2.imread(images[0],0)
img1 = cv2.imread(images[0],1)
img2 = cv2.imread(images[1],1)
# 标定
# 输入：世界坐标系里的位置 像素坐标 图像的像素尺寸大小 3*3矩阵，相机内参数矩阵 畸变矩阵
# 输出：标定结果 相机的内参数矩阵 畸变系数 旋转矩阵 平移向量
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
# mtx：内参数矩阵
# dist：畸变系数
# rvecs：旋转向量 （外参数）
# tvecs ：平移向量 （外参数）
print(("ret:"), ret)
print(("mtx:\n"), mtx)  # 内参数矩阵
print(("dist:\n"), dist)  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
print(("rvecs:\n"), rvecs)  # 旋转向量  # 外参数
print(("tvecs:\n"), tvecs)  # 平移向量  # 外参数

print(len(objpoints),len(imgpoints))
#双目标定
retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = \
    cv2.stereoCalibrate([objpoints[0]], [imgpoints[0]], [imgpoints[1]], mtx, dist, mtx, dist, gray.shape[::-1])   #再做双目标定


(R_l, R_r, P_l, P_r, Q, validPixROI1, validPixROI2) = \
    cv2.stereoRectify(mtx, dist, mtx, dist, gray.shape[::-1], R ,T)    #cv2.Rodrigues(rvecs[0]-rvecs[1])[0] # 计算旋转矩阵和投影矩阵

(map1, map2) = \
    cv2.initUndistortRectifyMap(mtx, dist, R_l, P_l, gray.shape[::-1],
                                cv2.CV_32FC1)  # 计算校正查找映射表

rect_left_image = cv2.remap(img1, map1, map2, cv2.INTER_CUBIC)  # 重映射

# 左右图需要分别计算校正查找映射表以及重映射
(map1, map2) = \
    cv2.initUndistortRectifyMap(mtx, dist, R_r, P_r, gray.shape[::-1], cv2.CV_32FC1)

rect_right_image = cv2.remap(img2, map1, map2, cv2.INTER_CUBIC)

imgcat_out = cat2images(rect_left_image, rect_right_image)
cv2.imshow("hello",rect_right_image)
cv2.waitKey()
cv2.imwrite('E:\\Dateset\\CameraCalibrationDataset-main\\imgcat_out.jpg', imgcat_out)




#计算重投影误差
# total_error = 0
#
# for i in range(len(objpoints)):
#     img = cv2.imread(images[i])
#     imgpoints1, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
#     error = cv2.norm(imgpoints[i], imgpoints1, cv2.NORM_L2) / len(imgpoints1)
#     total_error += error
# print(("total error: "), total_error / len(objpoints))

# 去畸变
# imgs = glob.glob('C:\\Users\\8389\\Downloads\\CameraCalibrationDataset-main\\CameraCalibrationDataset-main\\GoPro Hero 4\\*.jpg')
# i = 0
# for frame in imgs:
#     i += 1
#     img2 = cv2.imread(frame)
#     h, w = img2.shape[:2]
#     newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
#     dst = cv2.undistort(img2, mtx, dist, None, newcameramtx)
#     cv2.imwrite('C:\\Users\\8389\\Downloads\\CameraCalibrationDataset-main\\CameraCalibrationDataset-main\\GoPro Hero 4\\rect\\' + str(i) + '.jpg', dst)