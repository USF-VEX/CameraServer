import cv2
import numpy as np
import pyrealsense2 as rs
import torch
from pathlib import Path

from models.common import DetectMultiBackend
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_boxes
from utils.dataloaders import letterbox
from utils.torch_utils import select_device
from utils.plots import Annotator, colors, save_one_box
import math
import platform
import os
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

DEFAULT_SIZE = 480

def align_cameras(global_shutter_mtx, depth_sense_mtx, global_shutter_dist, depth_sense_dist, R, T, image_dimensions):
    width, height = image_dimensions
    # Calculate rotation and translation matrices for aligning the global shutter camera and the DepthSense camera
    R1, R2, P1, P2, Q, valid_roi1, valid_roi2 = cv2.stereoRectify(global_shutter_mtx, global_shutter_dist, depth_sense_mtx, depth_sense_dist, (width, height), R, T)

    global_shutter_map1, global_shutter_map2 = cv2.initUndistortRectifyMap(global_shutter_mtx, global_shutter_dist, R1, P1, (width, height), cv2.CV_16SC2)
    depth_sense_map1, depth_sense_map2 = cv2.initUndistortRectifyMap(depth_sense_mtx, depth_sense_dist, R2, P2, (width, height), cv2.CV_16SC2)

    return global_shutter_map1, global_shutter_map2, depth_sense_map1, depth_sense_map2

def image(camera):
    ret, frame = camera.read()
    return cv2.resize(frame, (DEFAULT_SIZE, DEFAULT_SIZE), interpolation=cv2.INTER_LINEAR)

def convert_yolo_detections_to_camera_coordinates(yolo_detections):
    camera_coordinates = []
    for detection in yolo_detections:
        xyxy, conf, cls = detection
        x1, y1, x2, y2 = xyxy
        width = x2 - x1
        height = y2 - y1
        x_center = x1 + width / 2
        y_center = y1 + height / 2
        camera_coordinates.append((x_center, y_center, width, height))
    return camera_coordinates

import numpy as np

def project_yolo_detections_to_depth_sense_camera_coordinates(yolo_detections_camera_coordinates, depth_map, depth_scale, depth_intrinsic):
    depth_sense_coordinates = []
    for detection in yolo_detections_camera_coordinates:
        x_left, y_top, width, height = detection
        x_center = x_left + width / 2
        y_center = y_top + height / 2

        print(depth_map)

        # Get the depth value at the center of the bounding box
        z_depth = depth_map[int(y_center), int(x_center)] * depth_scale

        # Convert pixel coordinates to 3D coordinates
        x_depth = (x_center - depth_intrinsic[0, 2]) * z_depth / depth_intrinsic[0, 0]
        y_depth = (y_center - depth_intrinsic[1, 2]) * z_depth / depth_intrinsic[1, 1]

        depth_sense_coordinates.append((x_depth, y_depth, z_depth))
    return depth_sense_coordinates

def extract_depth_information(yolo_detections_depth_sense_coordinates, depth_map):
    depth_information = []
    for coords in yolo_detections_depth_sense_coordinates:
        x_depth, y_depth, z_depth = coords
        depth_value = depth_map[int(y_depth), int(x_depth)]
        depth_information.append(depth_value)
    return depth_information

def calculate_object_position(yolo_detections, depth_information, camera_focal_length, image_width, image_height):
    object_positions = []
    for i, detection in enumerate(yolo_detections):
        xyxy, conf, cls = detection
        x1, y1, x2, y2 = xyxy
        width = x2 - x1
        height = y2 - y1
        x_center = x1 + width / 2
        y_center = y1 + height / 2
        depth_value = depth_information[i]

        # Calculate the X, Y coordinates
        x = (x_center - image_width / 2) * depth_value / camera_focal_length
        y = (y_center - image_height / 2) * depth_value / camera_focal_length

        # Calculate the angle
        angle = math.atan2(x, depth_value) * 180 / math.pi

        object_positions.append({'x': x, 'y': y, 'depth': depth_value, 'angle': angle})
    return object_positions



def preprocess_image(image, img_size=DEFAULT_SIZE):
    img = letterbox(image, img_size, stride=32)[0]
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().div(255.0)

def process_frame(frame, model, device, img_size=DEFAULT_SIZE, conf_thres=0.25, iou_thres=0.45):

    # Preprocess the frame
    input_tensor = preprocess_image(frame, img_size=img_size).to(device)
    
    # Perform the inference
    with torch.no_grad():
        pred = model(input_tensor, augment=False, visualize=False)[0]

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, 1000)

    # Process the detections
    detections = []
    
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            det[:, :4] = scale_boxes(input_tensor.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in det:
                detections.append((xyxy, conf, cls))

    return detections

def postprocess(frame, detections, model, object_positions = None):
    annotator = Annotator(frame, line_width=3, example=str(model.names))
    i = 0
    for *xyxy, conf, cls in detections:
        c = int(cls)
        label = f"{model.names[int(cls)]}: {conf:.2f}"
        xyxy = xyxy[0]
        annotator.box_label(xyxy, label, color=colors(c, True))
        if object_positions:
            pos = object_positions[i]
            xy_str = f"X: {pos['x']:.2f}, Y: {pos['y']:.2f}, Z: {pos['depth']:.2f}"
            x1, y1, _, _ = map(int, xyxy)
            cv2.putText(annotator.result(), xy_str, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        i+=1
    
    annotator.result()

# Add this at the beginning of your script
import json

# Load calibration data from JSON files
with open('global_shutter_calibration_data.json', 'r') as f:
    global_shutter_calib = json.load(f)

loaded_camera_matrix_gs = np.array(global_shutter_calib['camera_matrix'])
loaded_dist_coeffs_gs = np.array(global_shutter_calib['dist_coeffs'])

with open('depth_sense_calibration_data.json', 'r') as f:
    depth_sense_calib = json.load(f)

loaded_camera_matrix_ds = np.array(depth_sense_calib['camera_matrix'])
loaded_dist_coeffs_ds = np.array(depth_sense_calib['dist_coeffs'])

with open('stereo_calib_results.json', 'r') as f:
    stereo_calib_results = json.load(f)

R = np.array(stereo_calib_results["rotation_matrix"])
T = np.array(stereo_calib_results["translation_vector"])

width = 480
height = 480

global_shutter_camera = cv2.VideoCapture("tcp://192.168.1.92:8798")
pipeline = rs.pipeline()

# Configure streams
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)

# Start streaming
pipeline.start(config)

def read_depthsense_color_frame(self):
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_depth_frame()
    color_image = np.asanyarray(color_frame.get_data())
    return True, color_image

depth_sense_camera = type("", (), {"read": read_depthsense_color_frame})()

# Add this line after starting the pipeline
profile = pipeline.get_active_profile()
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# Modify the main function to use the created camera objects
def main():
    global DEFAULT_SIZE, DEFAULT_SIZE, depth_scale
    # Remove the calibrate_global_shutter_camera() and calibrate_depth_sense_camera() function calls

    # Align the cameras, if necessary
    global_shutter_map1, global_shutter_map2, depth_sense_map1, depth_sense_map2 = align_cameras(loaded_camera_matrix_gs, loaded_camera_matrix_ds, loaded_dist_coeffs_gs, loaded_dist_coeffs_ds, R, T, (DEFAULT_SIZE, DEFAULT_SIZE))

    weights = "E:/Jetson/best.pt" if platform.system() == "Windows" else "/home/robot/Jetson/best.pt"
    device = select_device('cpu') if platform.system() == "Windows" else select_device() # use 'cpu' if GPU is not available
    model = DetectMultiBackend(weights, device=device, dnn=False, data=ROOT / 'data/coco128.yaml', fp16=False)
    model.eval()

    model.warmup(imgsz=(1 if model.pt or model.triton else 1, 3, *(DEFAULT_SIZE, DEFAULT_SIZE)))

    while True:
        # Capture images from the global shutter camera and the DepthSense camera
        global_shutter_image = image(global_shutter_camera)
        depth_sense_depth_map = image(depth_sense_camera)

        # Remap the images using the maps obtained from align_cameras
        global_shutter_image_rectified = cv2.remap(global_shutter_image, global_shutter_map1, global_shutter_map2, cv2.INTER_LINEAR)
        depth_sense_depth_map_rectified = cv2.remap(depth_sense_depth_map, depth_sense_map1, depth_sense_map2, cv2.INTER_LINEAR)

        # Detect objects in the rectified global shutter camera image using YOLOv5
        yolo_detections = process_frame(global_shutter_image_rectified, model, device)

        # Convert YOLOv5 detections to global shutter camera coordinates
        yolo_detections_camera_coordinates = convert_yolo_detections_to_camera_coordinates(yolo_detections)

        # Project YOLOv5 detections to DepthSense camera coordinates
        yolo_detections_depth_sense_coordinates = project_yolo_detections_to_depth_sense_camera_coordinates(yolo_detections_camera_coordinates, depth_sense_depth_map_rectified, depth_scale, loaded_camera_matrix_ds)

        # Extract depth information for the detected objects
        depth_information = extract_depth_information(yolo_detections_depth_sense_coordinates, depth_sense_depth_map_rectified)

        # Calculate the X, Y coordinates and angle of detected objects from the camera
        object_positions = calculate_object_position(yolo_detections, depth_information, loaded_camera_matrix_gs[0, 0], width, height)

        frame = postprocess(global_shutter_image_rectified, yolo_detections, model, object_positions)

        # print(img)

        # Process object_positions as needed (e.g., display or store the results)
        try:
            cv2.imshow("Global Shutter Image", global_shutter_image_rectified)
        except:
            pass
        cv2.waitKey(1)

import sys

if __name__ == "__main__":
    sys.exit(main())