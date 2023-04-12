import cv2
import numpy as np

filename = "C:\\Users\\8389\\Pictures\\work\\persptive.jpg"
img_gray = cv2.imread(filename,0)
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(img_gray)
# 输入图像必须是 float32 ,最后一个参数在 0.04 到 0.05 之间
dst = cv2.cornerHarris(gray, 2, 3, 0.04)
# result is dilated for marking the corners, not important
# dst = cv2.dilate(dst,None)
# Threshold for an optimal value, it may vary depending on the image.
img_gray[dst > 0.09 * dst.max()] = [0, 0, 255]

cv2.imshow('dst', img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()

coords = np.array(np.where(dst > 0.01 * dst.max()))

keypoints = []
if coords is not None and len(coords) > 0:
    for x, y in zip(coords[1], coords[0]):
        keypoints.append((x, y))

pts_d = np.float32([[0, 0], [0, 450], [1010, 450], [1010, 0]])
pts_0 = np.float32([[395,212], [428,916], [1506,743], [1500,189]])

cv2.circle(img,[395, 212],3,(255,0,0))

hull = cv2.convexHull(np.array((keypoints)))
#print(hull)
#print(hull.reshape(5, 2)[0:4, :])
M = cv2.getPerspectiveTransform(pts_0, pts_d)
print(M)
dst = cv2.warpPerspective(img, M, (1010, 450))

cv2.polylines(img, [hull], True, (255, 255, 0), 2)
cv2.imwrite('C:\\Users\\8389\\Pictures\\work\\persptive1.jpg', dst)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()