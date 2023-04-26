import numpy as np
import cv2
import glob
import json
import time

# Define the size of the checkerboard
# CHECKERBOARD = (8, 6)

# # Prepare object points
# objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
# objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# # Arrays to store object points and image points from all the images
# objpoints = []
# imgpoints1 = []
# imgpoints2 = []

# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# # Load images
# images1 = glob.glob('global_shutter_images/*.png')
# images2 = glob.glob('depth_sense_images/*.png')

# for i in range(len(images1)):
#     img1 = cv2.imread(images1[i])
#     img2 = cv2.imread(images2[i])

#     # Convert images to grayscale
#     gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#     gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#     # Find the chess board corners
#     ret1, corners1 = cv2.findChessboardCorners(gray1, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
#     ret2, corners2 = cv2.findChessboardCorners(gray2, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

#     if ret1 and ret2:
#         objpoints.append(objp)

#         # Refine corner location for accuracy
#         corners1 = cv2.cornerSubPix(gray1, corners1, (3, 3), (-1, -1), criteria)
#         imgpoints1.append(corners1)

#         corners2 = cv2.cornerSubPix(gray2, corners2, (3, 3), (-1, -1), criteria)
#         imgpoints2.append(corners2)

# dt = time.time()
# # Calibrate each camera separately using the detected corners
# ret1, mtx1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints, imgpoints1, gray1.shape[::-1], None, None)
# ret2, mtx2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(objpoints, imgpoints2, gray2.shape[::-1], None, None)
# print("Calibration Time:", time.time() - dt)

# # Save the calibration parameters to a JSON file
# calibration_params = {
#     'mtx1': mtx1.tolist(),
#     'dist1': dist1.tolist(),
#     'mtx2': mtx2.tolist(),
#     'dist2': dist2.tolist()
# }

# with open('calibration_params.json', 'w') as f:
#     json.dump(calibration_params, f)

# Load the calibration parameters from the JSON file
with open('calibration_params.json', 'r') as f:
    calibration_params = json.load(f)

mtx1 = np.array(calibration_params['mtx1'])
dist1 = np.array(calibration_params['dist1'])
mtx2 = np.array(calibration_params['mtx2'])
dist2 = np.array(calibration_params['dist2'])

# Load the images
global_shutter_img = cv2.imread('global_shutter_images/global_shutter_frame_0.png')
depth_mask_img = cv2.imread('depth_sense_images/depth_sense_frame_0.png')

# Define the zoom level as a scaling factor
zoom = 1.6
factor_h = 485
factor_w = 205

while True:
    dt = time.time()
    # Undistort the images
    global_shutter_img_undistorted = cv2.undistort(global_shutter_img, mtx1, dist1)
    depth_mask_img_undistorted = cv2.undistort(depth_mask_img, mtx2, dist2)

    # Resize the image to zoom in
    depth_mask_img_undistorted = cv2.resize(depth_mask_img_undistorted, None, fx=zoom, fy=zoom+.5, interpolation=cv2.INTER_LINEAR)

    # Crop the center of the image to 640x640
    h, w, _ = depth_mask_img_undistorted.shape

    crop_start_h = (h - factor_h) // 2
    crop_start_w = (h - factor_w) // 2
    depth_mask_img_undistorted = depth_mask_img_undistorted[crop_start_h:crop_start_h+480, crop_start_w:crop_start_w+480]

    try:
        # Combine the images
        combined_img = cv2.addWeighted(global_shutter_img_undistorted, 0.5, depth_mask_img_undistorted, 0.5, 0)
        print("Restore Time:", time.time() - dt)
    except:
        factor_h += 1
        factor_w += 1
        #zoom += 0.01
        #zoom = max(zoom, 1.5)
        cv2.waitKey(1)
        continue

    # Display the combined image
    cv2.imshow('Combined Image', combined_img)
    cv2.imshow('Global Image', global_shutter_img_undistorted)
    cv2.imshow('Intel Image', depth_mask_img_undistorted)
    key = cv2.waitKey(0)

    if key == ord('h'):
        factor_h += 10
    if key == ord('b'):
        factor_h -= 10
    if key == ord('w'):
        factor_w += 10
    if key == ord('s'):
        factor_w -= 10
    if key == ord('c'):
        zoom += 0.05
    if key == ord('v'):
        zoom -= 0.05

    print("zoom", zoom)
    print("factor_h", factor_h)
    print("factor_w", factor_w)


cv2.destroyAllWindows()