import pyrealsense2 as rs

ctx = rs.context()
devices = ctx.query_devices()

for dev in devices:
    print(f"Device: {dev.get_info(rs.camera_info.name)}")
    for sensor in dev.query_sensors():
        print(f"  Sensor: {sensor.get_info(rs.camera_info.name)}")
        for stream in sensor.get_stream_profiles():
            print(f"    Stream: {stream}")