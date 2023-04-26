import cv2
import numpy as np
import os
import pyrealsense2 as rs

DEFAULT_SIZE = 480

FRAME = None

def calibrate_camera(video_source, checkerboard_size, square_size, camera_name):
    global FRAME
    # Termination criteria for the iterative optimization algorithm
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0), ..., (5,8,0)
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2) * square_size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.
    pipeline = None

    if video_source == 0:
        pipeline = rs.pipeline()

        # Configure streams
        config = rs.config()
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)

        # Start streaming
        pipeline.start(config)

        def read_depthsense_color_frame(self):
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            return True, color_image

        cam = type("", (), {"read": read_depthsense_color_frame})()
    else:
        cam = cv2.VideoCapture(video_source)
        def tmp(self):
            return FRAME
        cam = type("", (), {"read": tmp})()

    frame_count = 100

    if not os.path.exists(f"{camera_name}_images"):
        os.makedirs(f"{camera_name}_images")

    while True:
        orignal_frame = None
        ret, orignal_frame = cam.read()
        if not ret:
            continue

        orignal_frame = cv2.resize(orignal_frame, (DEFAULT_SIZE, DEFAULT_SIZE), interpolation=cv2.INTER_LINEAR)

        gray = cv2.cvtColor(orignal_frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

        _frame = None

        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            frame_count -= 1
            cv2.imwrite(f"{camera_name}_images/{camera_name}_frame_{frame_count}.png", orignal_frame)

            # Draw and display the corners
            _frame = cv2.drawChessboardCorners(orignal_frame, checkerboard_size, corners2, ret)

        cv2.imshow("Frame", orignal_frame if _frame is None else _frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("c") or frame_count <= 0:
            break

    if pipeline is not None:
        pipeline.stop()

    cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return ret, mtx, dist, rvecs, tvecs

checkerboard_size = (8, 6)  # Number of internal corners in the checkerboard pattern (width, height)
square_size = 1.2 * 25.4  # Square size in the checkerboard (in your case, 1.2 inch or 30.48 mm)

video_source_gs = "tcp://192.168.1.92:8798"
video_source_ds = 0

ret_gs, mtx_gs, dist_gs, rvecs_gs, tvecs_gs = calibrate_camera(video_source_gs, checkerboard_size, square_size, "global_shutter")
ret_ds, mtx_ds, dist_ds, rvecs_ds, tvecs_ds = calibrate_camera(video_source_ds, checkerboard_size, square_size, "depth_sense")

# print("Global shutter camera matrix:", mtx_gs)
# print("Global shutterdistortion coefficients:", dist_gs)
# print("Global shutter rotation vectors:", rvecs_gs)
# print("Global shutter translation vectors:", tvecs_gs)

# print("Depth sense camera matrix:", mtx_ds)
# print("Depth sense distortion coefficients:", dist_ds)
# print("Depth sense rotation vectors:", rvecs_ds)
# print("Depth sense translation vectors:", tvecs_ds)

import json

calibration_data_gs = {
'camera_matrix': mtx_gs.tolist(),
'dist_coeffs': dist_gs.tolist(),
'rvecs': [rvec.tolist() for rvec in rvecs_gs],
'tvecs': [tvec.tolist() for tvec in tvecs_gs]
}

calibration_data_ds = {
'camera_matrix': mtx_ds.tolist(),
'dist_coeffs': dist_ds.tolist(),
'rvecs': [rvec.tolist() for rvec in rvecs_ds],
'tvecs': [tvec.tolist() for tvec in tvecs_ds]
}

with open('global_shutter_calibration_data.json', 'w') as f:
    json.dump(calibration_data_gs, f)

with open('depth_sense_calibration_data.json', 'w') as f:
    json.dump(calibration_data_ds, f)