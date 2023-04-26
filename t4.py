import subprocess
import cv2
import numpy as np
import platform

# Set up FFmpeg command
input_stream = 'tcp://192.168.1.92:8798'
path = "E:\\Desktop\\All Folders\\ffmpeg\\bin\\ffmpeg.exe" if platform.system() == "Windows" else "ffmpeg"
ffmpeg_command = f'{path} -i {input_stream} -vf scale=1456:800 -pix_fmt bgr24 -vcodec rawvideo -f image2pipe pipe:'.split()

# Set up FFmpeg process
proc = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

while True:
    # Read raw image bytes from FFmpeg process output
    raw_image = proc.stdout.read(1456 * 800 * 3)
    if not raw_image:
        continue

    # Convert raw bytes to numpy array
    image_array = np.frombuffer(raw_image, np.uint8)
    image = image_array.reshape((800, 1456, 3))

    # Display the image
    cv2.imshow('Video Stream', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()
proc.kill()
