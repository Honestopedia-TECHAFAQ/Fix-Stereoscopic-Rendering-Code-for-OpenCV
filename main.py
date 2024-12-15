import cv2
import numpy as np

def load_images(left_image_path, right_image_path):
    left_image = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
    right_image = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)
    return left_image, right_image

def stereo_rectification(left_image, right_image):
    camera_matrix = np.eye(3)
    dist_coeffs = np.zeros((5, 1))
    R = np.eye(3)
    T = np.array([1, 0, 0], dtype=np.float32)
    image_size = (left_image.shape[1], left_image.shape[0])
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(camera_matrix, dist_coeffs, camera_matrix, dist_coeffs, image_size, R, T, alpha=0)
    map1x, map1y = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, R1, P1, image_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, R2, P2, image_size, cv2.CV_32FC1)
    rectified_left = cv2.remap(left_image, map1x, map1y, cv2.INTER_LINEAR)
    rectified_right = cv2.remap(right_image, map2x, map2y, cv2.INTER_LINEAR)
    return rectified_left, rectified_right, Q

def compute_depth_map(left_image, right_image):
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16 * 5,
        blockSize=11,
        P1=8 * 3 * 11**2,
        P2=32 * 3 * 11**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )
    disparity = stereo.compute(left_image, right_image).astype(np.float32) / 16.0
    return disparity

def generate_3d_points(disparity, Q):
    points_3D = cv2.reprojectImageTo3D(disparity, Q)
    return points_3D

left_image_path = "path_to_left_image.png"
right_image_path = "path_to_right_image.png"

left_image, right_image = load_images(left_image_path, right_image_path)
rectified_left, rectified_right, Q = stereo_rectification(left_image, right_image)
disparity = compute_depth_map(rectified_left, rectified_right)
points_3D = generate_3d_points(disparity, Q)

cv2.imshow("Disparity", (disparity - disparity.min()) / (disparity.max() - disparity.min()))
cv2.waitKey(0)
cv2.destroyAllWindows()
