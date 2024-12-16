import os.path

import cv2
import numpy as np
import pyrealsense2 as rs

from utils.robot_state import RobotState

def collect_data(data_folder):
    """
    Collect data for camera calibration, save the images and the TCP poses to the data folder
    :param data_folder: str, the folder to save the data
    :return: None
    """
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        os.makedirs(os.path.join(data_folder, "images"))
        os.makedirs(os.path.join(data_folder, "tcp_poses"))

    # init the robot connection
    robot_state = RobotState()

    pattern_size = (9, 7)

    # Initialize RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

    # Start the pipeline
    pipeline.start(config)

    print("Press Enter to save an image and Esc to stop.")

    frame_count = 0

    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            # Convert to numpy array
            color_image = np.asanyarray(color_frame.get_data())

            draw_img = color_image.copy()

            gray = cv2.cvtColor(draw_img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, pattern_size)
            cv2.drawChessboardCorners(draw_img, pattern_size, corners, ret)

            # Display the image
            cv2.imshow('RealSense', draw_img)

            # Check for key presses
            key = cv2.waitKey(1) & 0xFF

            if key == 13:  # Enter key to save image
                cv2.imwrite(os.path.join(data_folder, "images", f'frame_{frame_count:04d}.png'), color_image)
                print(f"Saved frame_{frame_count:04d}.png")
                tcp_mat = robot_state.get_tcp_mat()
                np.save(os.path.join(data_folder, "tcp_poses", f'frame_{frame_count:04d}_tcp.npy'), tcp_mat)
                print(f"Saved frame_{frame_count:04d}_tcp.npy")
                frame_count += 1

            if key == 27:  # Escape key to exit
                print("Exiting...")
                break

    finally:
        # Stop the pipeline and close OpenCV windows
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    data_folder = "/home/yunlongwang/workspace/cam_cali_ws/data"
    collect_data(data_folder)