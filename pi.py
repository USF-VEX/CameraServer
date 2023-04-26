import socket
import platform
import subprocess as ps
import numpy as np
import atexit
from picamera import PiCamera
import io

# picamera = "libcamera-vid --nopreview --shutter 2000 -t 0 --width 1456 --height 1088 --gain 20 --framerate 60 --denoise cdn_fast --rawfull 1 --exposure sport --flush 0 -o -".split()

server_ip = "192.168.1.92" if platform.system() == "Linux" else "192.168.1.3"
server_port = 8798

width = 1456
height = 1088
bytespframe = int(1456 * 1088 * 3)

picam = PiCamera(resolution=(width, height), framerate=60)
picam.shutter_speed = 2000
picam.exposure_mode = 'off'
print(picam)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((server_ip, server_port))
    sock.listen()

    print("Waiting for client to connect...")
    conn, addr = sock.accept()
    print("Client connected:", addr)

    # camera_process = ps.Popen(picamera, stdout=ps.PIPE)  # start the camera
    # atexit.register(camera_process.terminate)

    # while True:
        
        # frame = np.frombuffer(camera_process.stdout.read(bytespframe), dtype=np.uint8)

        # print(frame)

    stream = io.BytesIO()
    for _ in picam.capture_continuous(stream, 'jpeg'):
        frame = stream.read()
        # Reset the stream for the next capture
        stream.seek(0)
        stream.truncate()

        if frame.size != bytespframe:
            # print("Incomplete frame, skipping")
            continue
        try:
            # Send the bytes over the network
            conn.sendall(frame)
        except (BrokenPipeError, ConnectionResetError) as  e:
            print("Client Disconnected: Waiting for client to connect...")
            conn.close()
            conn, addr = sock.accept()
            print("Client connected:", addr)
        except KeyboardInterrupt:
            print("Stopping server...")
            break
        except:
            print("Server Error")
            break

    conn.close()

# import io
# import socket
# import struct
# import time
# import picamera
# from multiprocessing.connection import Client
# ROBOT_SERVER = ('192.168.1.3', 8798)

# # Connect a client socket to 192.168.1.3:8798 (change my_server to the
# # hostname of your server)
# client = Client(ROBOT_SERVER, authkey=b'CAMERA0',)

# try:
#     camera = picamera.PiCamera()
#     camera.resolution = (1456, 1088)
#     camera.shutter_speed = 2000
#     # Start a preview and let the camera warm up for 2 seconds
#     camera.start_preview()
#     time.sleep(2)

#     start = time.time()
#     stream = io.BytesIO()
#     for _ in camera.capture_continuous(stream, 'jpeg'):
#         client.send(stream.read())
#         # Reset the stream for the next capture
#         stream.seek(0)
#         stream.truncate()
# finally:
#     client.close()