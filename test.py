import cv2
import numpy as np
import pyrealsense2 as rs
import threading
import queue
import sys
import platform
import subprocess

def read_frames(process, width, height, frame_queue):
    while True:
        error_line = process.stderr.readline()
        if error_line:
            print("Error: ", error_line)
        raw_image_data = process.stdout.read(width * height * 3)
        if not raw_image_data:
            continue
        try:
            # Convert the raw image data to a NumPy array
            image_array = np.frombuffer(raw_image_data, dtype=np.uint8).reshape((height, width, 3))
            frame_queue.put(image_array)
        except:
            pass    

global_frame = None
DEFAULT_SIZE = 480

def getCam(video_source):
    global global_frame
    if video_source == 0:
        pipeline = rs.pipeline()

        # Configure streams
        config = rs.config()
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)

        # Start streaming
        pipeline.start(config)

        def read_depthsense_color_frame(self):
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            return True, color_image

        return type("", (), {"read": read_depthsense_color_frame, "pipeline": pipeline})()
    else:

        # Set up FFmpeg command
        input_stream = 'tcp://192.168.1.92:8798'
        path = "E:\\Desktop\\All Folders\\ffmpeg\\bin\\ffmpeg.exe" if platform.system() == "Windows" else "ffmpeg"
        ffmpeg_command = f'{path} -i {input_stream} -vf scale=1456:800 -fflags nobuffer -flags low_delay -vsync 2 -preset ultrafast -pix_fmt bgr24 -vcodec rawvideo -f image2pipe pipe:'.split()

        # Set up FFmpeg process
        process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        def read_frames(proc, frame_queue: queue.Queue):
            global global_frame
            while True:
                # Read raw image bytes from FFmpeg process output
                raw_image = proc.stdout.read(1456 * 800 * 3)
                if not raw_image:
                    continue

                # Convert raw bytes to numpy array
                image_array = np.frombuffer(raw_image, np.uint8)
                global_frame = image_array.reshape((800, 1456, 3))

        frame_queue = queue.Queue(maxsize=5)
        reader_thread = threading.Thread(target=read_frames, args=(process, frame_queue))
        reader_thread.start()

        def getFrame(self):
            global global_frame
            if global_frame is not None:
                return True, global_frame
            else:
                return False, None

        return type("", (), {"read": getFrame, "frame_queue": frame_queue, "reader_thread": reader_thread})()

import os

def calibrate_camera(video_source, checkerboard_size, square_size, camera_name):
    global global_frame
    # Termination criteria for the iterative optimization algorithm
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0), ..., (5,8,0)
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2) * square_size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.
    pipeline = None

    cam = getCam(video_source)

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

    # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    # return ret, mtx, dist, rvecs, tvecs

# video_source_gs = "tcp://192.168.1.92:8798"
# video_source_ds = 0

# cap1 = getCam(video_source_gs)
# cap2 = getCam(video_source_ds)

checkerboard_size = (8, 6)  # Number of internal corners in the checkerboard pattern (width, height)
square_size = 1.2 * 25.4  # Square size in the checkerboard (in your case, 1.2 inch or 30.48 mm)

video_source_gs = "tcp://192.168.1.92:8798"
video_source_ds = 0

calibrate_camera(video_source_gs, checkerboard_size, square_size, "global_shutter")
calibrate_camera(video_source_ds, checkerboard_size, square_size, "depth_sense")

import json

# calibration_data_gs = {
# 'camera_matrix': mtx_gs.tolist(),
# 'dist_coeffs': dist_gs.tolist(),
# 'rvecs': [rvec.tolist() for rvec in rvecs_gs],
# 'tvecs': [tvec.tolist() for tvec in tvecs_gs]
# }

# calibration_data_ds = {
# 'camera_matrix': mtx_ds.tolist(),
# 'dist_coeffs': dist_ds.tolist(),
# 'rvecs': [rvec.tolist() for rvec in rvecs_ds],
# 'tvecs': [tvec.tolist() for tvec in tvecs_ds]
# }

# with open('global_shutter_calibration_data.json', 'w') as f:
#     json.dump(calibration_data_gs, f)

# with open('depth_sense_calibration_data.json', 'w') as f:
    # json.dump(calibration_data_ds, f)

# while True:
#     # Read frames from the global shutter camera queue (non-blocking)
#     # ret1, frame1 = cap1.read()

#     # # Read frames from the second video feed
#     ret2, frame2 = cap2.read()
    
#     # # Check if the frame from the second video feed was read successfully
#     if global_frame is None or not ret2:
#         continue

#     # Resize frames
#     frame1 = cv2.resize(global_frame, (640, 640))
#     frame2 = cv2.resize(frame2, (640, 640))

#     # Overlay the frames with 50% transparency each
#     # blended_frame = cv2.addWeighted(frame1, 0.5, frame2, 0.5, 0)

#     # Display the blended frame
#     # cv2.imshow('Blended Video', blended_frame)

#     # cv2.imshow('Global', frame1)
#     # cv2.imshow('Depth', frame2)

#     # Exit if the user presses the 'q' key
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break