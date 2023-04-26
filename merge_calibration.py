import cv2
import numpy as np
import glob
import json
from pathlib import Path

# Checkerboard pattern dimensions and square size
pattern_size = (8, 6)  # Number of inner corners in the checkerboard pattern
square_size = 1.2 * 25.4  # Size of each square in the checkerboard pattern (in cm)

def load_image_files(folder):
    image_files = [str(file) for file in Path(folder).rglob("*.png")]
    image_files.sort()
    return image_files

with open("global_shutter_calibration_data.json", "r") as f:
    global_shutter_calib = json.load(f)

    global_shutter_mtx = np.array(global_shutter_calib["camera_matrix"])
    global_shutter_dist = np.array(global_shutter_calib["dist_coeffs"])

with open("depth_sense_calibration_data.json", "r") as f:
    depth_sense_calib = json.load(f)

    depth_sense_mtx = np.array(depth_sense_calib["camera_matrix"])
    depth_sense_dist = np.array(depth_sense_calib["dist_coeffs"])

# Load the pairs of images captured from both cameras
global_shutter_images = load_image_files('global_shutter_images')
depth_sense_images = load_image_files('depth_sense_images')

global_shutter_imgs = [cv2.imread(img_path) for img_path in global_shutter_images]
depth_sense_imgs = [cv2.imread(img_path) for img_path in depth_sense_images]

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0), ..., (8,5,0)
objp = np.zeros((np.prod(pattern_size), 3), np.float32)
objp[:, :2] = np.indices(pattern_size).T.reshape(-1, 2) * square_size

object_points = []
global_shutter_image_points = []
depth_sense_image_points = []

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.001)

for gs_img, ds_img in zip(global_shutter_imgs, depth_sense_imgs):
    gs_gray = cv2.cvtColor(gs_img, cv2.COLOR_BGR2GRAY)
    ds_gray = cv2.cvtColor(ds_img, cv2.COLOR_BGR2GRAY)

    gs_ret, gs_corners = cv2.findChessboardCorners(gs_gray, pattern_size)
    ds_ret, ds_corners = cv2.findChessboardCorners(ds_gray, pattern_size)

    if gs_ret and ds_ret:
        object_points.append(objp)

        gs_corners_refined = cv2.cornerSubPix(gs_gray, gs_corners, (11, 11), (-1, -1), criteria)
        global_shutter_image_points.append(gs_corners_refined)

        ds_corners_refined = cv2.cornerSubPix(ds_gray, ds_corners, (11, 11), (-1, -1), criteria)
        depth_sense_image_points.append(ds_corners_refined)

retval, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
    object_points,
    global_shutter_image_points,
    depth_sense_image_points,
    global_shutter_mtx,
    global_shutter_dist,
    depth_sense_mtx,
    depth_sense_dist,
    global_shutter_imgs[0].shape[::-1][1:],
    flags=cv2.CALIB_FIX_INTRINSIC
)

stereo_calib_results = {
    "rotation_matrix": R.tolist(),
    "translation_vector": T.tolist()
}

with open("stereo_calib_results.json", "w") as f:
    json.dump(stereo_calib_results, f)

print("Stereo calibration complete.")