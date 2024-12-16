import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

class CameraCalibration:
    """
    This class is used to calibrate the camera and the robot's TCP poses
    """
    def __init__(self,
        image_folder,
        Transforms_folder,
        pattern_size=(9, 7),
        square_size=20/1000,
        ShowProjectError=False,
        ShowCorners=False,
     ):
        """
        :param image_folder: str, the folder containing the images
        :param Transforms_folder: str, the folder containing the TCP poses
        :param pattern_size: tuple, the size of the chessboard pattern
        :param square_size: float, the size of the squares in the chessboard pattern
        :param ShowProjectError: bool, if True, the projection error will be shown
        :param ShowCorners: bool, if True, the images with the detected corners will be shown
        """

        # Initiate parameters
        self.pattern_size = pattern_size
        self.square_size = square_size
        self.ShowCorners = ShowCorners
        self.ShowProjectError = ShowProjectError
        self.ws_root = os.path.dirname(image_folder)

        # load images and TCP poses
        self.image_files = sorted(glob.glob(f'{image_folder}/*.png'))
        self.transform_files = sorted(glob.glob(f'{Transforms_folder}/*.npy'))
        self.images = [cv2.imread(f) for f in self.image_files]
        self.All_T_base2EE_list = [np.load(f) for f in self.transform_files]

    # run the calibration
    def run(self):
        # find chessboard corners and filter out the images without corners
        self.chessboard_corners, self.IndexWithImg = self.find_chessboard_corners(self.images, self.pattern_size, ShowCorners=self.ShowCorners)

        # calibrate the camera from the detected corners
        self.intrinsic_matrix, self.distr_coeff, self.rvecs, self.tvecs = self.calibrate_camera(
            self.chessboard_corners,
            self.IndexWithImg,
            self.pattern_size,
            self.square_size,
            self.images[0].shape[:2],
            ShowProjectError=self.ShowProjectError
        )

        # Extract the corresponding TCP poses
        self.TBase2EE = [self.All_T_base2EE_list[i] for i in self.IndexWithImg]

        #Calculate camera extrinsics
        self.RTarget2Cam, self.tTarget2Cam = self.compute_camera_poses(self.chessboard_corners,
                                                                       self.pattern_size, self.square_size,
                                                                       self.intrinsic_matrix,
                                                                       self.distr_coeff)

        #Convert to homogeneous transformation matrix
        self.T_target2cam = [np.concatenate((R, T), axis=1) for R, T in zip(self.RTarget2Cam, self.tTarget2Cam)]
        for i in range(len(self.T_target2cam)):
            self.T_target2cam[i] = np.concatenate((self.T_target2cam[i], np.array([[0, 0, 0, 1]])), axis=0)

        #Calculate T_cam2target
        self.T_cam2target = [np.linalg.inv(T) for T in self.T_target2cam]
        self.RCam2Target = [T[:3, :3] for T in self.T_cam2target]
        self.tCam2Target = [np.expand_dims(T[:3, 3], axis=-1) for T in self.T_cam2target]

        # Calculate related transforms
        self.RBase2EE = [T[:3, :3] for T in self.TBase2EE]
        self.tBase2EE = [np.expand_dims(T[:3, 3], axis=-1) for T in self.TBase2EE]
        self.TEE2Base = [np.linalg.inv(T) for T in self.TBase2EE]
        self.REE2Base = [T[:3, :3] for T in self.TEE2Base]
        self.tEE2Base = [np.expand_dims(T[:3, 3], axis=-1) for T in self.TEE2Base]

        #Create folder to save final transforms
        transform_folder = os.path.join(self.ws_root, "FinalTransforms")
        if not os.path.exists(transform_folder):
            os.mkdir(transform_folder)

        # solve hand-eye calibration
        for i in range(0, 5):
            print("Method:", i)
            self.R_cam2gripper, self.t_cam2gripper = cv2.calibrateHandEye(
                R_gripper2base=self.RBase2EE,
                t_gripper2base=self.tBase2EE,
                R_target2cam=self.RTarget2Cam,
                t_target2cam=self.tTarget2Cam,
                method=i
            )
            # print and save each results as .npz file
            print("The results for method", i, "are:")
            print("t_cam2gripper:", self.t_cam2gripper)
            print("R_cam2gripper:", self.R_cam2gripper)
            # Create 4x4 transfromation matrix
            self.T_cam2gripper = np.concatenate((self.R_cam2gripper, self.t_cam2gripper), axis=1)
            self.T_cam2gripper = np.concatenate((self.T_cam2gripper, np.array([[0, 0, 0, 1]])), axis=0)
            # Save results in folder FinalTransforms
            np.savetxt(os.path.join(transform_folder, f'T_cam2gripper_Method_{i}.txt'), self.T_cam2gripper,
                       fmt='%.4f', delimiter=',')
            # Save the inverse transform
            self.T_gripper2cam = np.linalg.inv(self.T_cam2gripper)
            np.savetxt(os.path.join(transform_folder, f'T_gripper2cam_Method_{i}.txt'), self.T_gripper2cam,
                       fmt='%.4f', delimiter=',')

    def find_chessboard_corners(self, images, pattern_size, ShowCorners=False):
        """
        Finds the corners of the chessboard in the images
        :param images: np.ndarray[], the images
        :param pattern_size: tuple, the size of the chessboard pattern
        :param ShowCorners: bool, if True, the images with the detected corners will be shown
        :return: the chessboard corners and the index of the images with detected corners
        """
        _folder = os.path.join(self.ws_root, "DetectedCorners")
        chessboard_corners = []
        IndexWithImg = []
        i = 0
        for image in tqdm(images, "Finding corners..."):
            _image = image.copy()
            gray = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, pattern_size)
            if ret:
                chessboard_corners.append(corners)

                cv2.drawChessboardCorners(_image, pattern_size, corners, ret)
                if ShowCorners:

                    plt.imshow(_image)
                    plt.title("Detected corner in image: " + str(i))
                    plt.show()

                if not os.path.exists(_folder):
                    os.makedirs(_folder)

                cv2.imwrite(os.path.join(_folder, "DetectedCorners" + str(i) + ".png"), _image)

                IndexWithImg.append(i)
                i = i + 1
            else:
                print("No chessboard found in image: ", i)
                i = i + 1
        return chessboard_corners, IndexWithImg

    def compute_camera_poses(self, chessboard_corners, pattern_size, square_size, intrinsic_matrix, dist_coeffs):
        """
        Computes the camera poses from the detected chessboard corners
        :param chessboard_corners: the corners of the chessboard
        :param pattern_size: tuple, the size of the chessboard pattern
        :param square_size: tuple, the size of the squares in the chessboard pattern
        :param intrinsic_matrix: matrix, the intrinsic matrix of the camera
        :param dist_coeffs: vector, the distortion coefficients of the camera
        :return:
        """
        # Create the object points.Object points are points in the real world that we want to find the pose of.
        object_points = np.zeros((pattern_size[0] * pattern_size[1], 3), dtype=np.float32)
        object_points[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

        # Estimate the pose of the chessboard corners
        RTarget2Cam = []
        TTarget2Cam = []
        i = 1
        for corners in tqdm(chessboard_corners, "Computing camera poses..."):
            _, rvec, tvec = cv2.solvePnP(object_points, corners, intrinsic_matrix, dist_coeffs)
            # rvec is the rotation vector, tvec is the translation vector
            i = 1 + i
            R, _ = cv2.Rodrigues(rvec)  # R is the rotation matrix from the target frame to the camera frame
            RTarget2Cam.append(R)
            TTarget2Cam.append(tvec)

        return RTarget2Cam, TTarget2Cam

    def calibrate_camera(self, chessboard_corners, IndexWithImg, pattern_size, square_size, ImgSize, ShowProjectError=False):
        """
        Calibrates the camera using the detected corners.
        NOTE: If using this function, keep the camera intrinsic matrix and distortion coefficients for future use.
        :param chessboard_corners: the corners of the chessboard
        :param IndexWithImg: index of the images with detected corners
        :param pattern_size: tuple, the size of the chessboard pattern
        :param square_size: float, the size of the squares in the chessboard pattern
        :param ImgSize: tuple, the size of the images
        :param ShowProjectError: bool, if True, the projection error will be shown
        :return: the intrinsic matrix, the distortion coefficients, the rotation vectors, and the translation vectors
        """
        # Find the corners of the chessboard in the image
        imgpoints = chessboard_corners
        # Find the corners of the chessboard in the real world
        objpoints = []
        for _ in range(len(IndexWithImg)):
            objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size
            objpoints.append(objp)
        # Find the intrinsic matrix
        print("Calculating intrinsics...")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, ImgSize, None, None)
        _folder = os.path.join(self.ws_root, "camera_calibration_results")
        if not os.path.exists(_folder):
            os.makedirs(_folder, exist_ok=True)
        np.savetxt(os.path.join(_folder, 'camera_matrix.txt'), mtx, delimiter=',', fmt='%f')
        print(f"Camera matrix: {mtx}")
        np.savetxt(os.path.join(_folder, 'distortion_coefficients.txt'), dist, delimiter=',', fmt='%f')
        print(f"Distortion coefficients: {dist}")

        self.extract_undistorted_images(mtx, dist)
        print("Calibration re-projection error:")
        print("The projection error from the calibration is: ",
              self.calculate_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist, ShowProjectError))
        return mtx, dist, rvecs, tvecs

    def extract_undistorted_images(self, intrinsic_matrix, distortion_coefficients):
        """
        Save the intermediate undistorted images for debugging
        :param intrinsic_matrix: the intrinsic matrix of the camera, 3x3 matrix
        :param distortion_coefficients: the distortion coefficients of the camera, 1x5 matrix
        :return:
        """
        output_folder = os.path.join(self.ws_root, "undistortedImages")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)
        for i in range(len(self.images)):
            img = self.images[i]
            h, w = img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(intrinsic_matrix, distortion_coefficients, (w, h), 1, (w, h))
            dst = cv2.undistort(img, intrinsic_matrix, distortion_coefficients, None, newcameramtx)
            x, y, w, h = roi
            dst = dst[y:y + h, x:x + w]
            cv2.imwrite(os.path.join(output_folder, f"frame_{i:04d}.png"), dst)

    def calculate_reprojection_error(self, objpoints, imgpoints, rvecs, tvecs, mtx, dist,ShowPlot=False):
        """
        Calculates the reprojection error of the camera for each image. The output is the mean reprojection error
        If ShowPlot is True, it will show the reprojection error for each image in a bar graph
        :param objpoints: the object points (x, y, z)
        :param imgpoints: the image points (u, v)
        :param rvecs: the rotation vectors of the camera
        :param tvecs: the translation vectors of the camera
        :param mtx: the intrinsic matrix of the camera
        :param dist: the distortion coefficients of the camera
        :param ShowPlot: bool, if True, the reprojection error will be shown in a bar graph
        :return:
        """

        total_error = 0
        num_points = 0
        errors = []

        for i in range(len(objpoints)):
            imgpoints_projected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            imgpoints_projected = imgpoints_projected.reshape(-1, 1, 2)
            error = cv2.norm(imgpoints[i], imgpoints_projected, cv2.NORM_L2) / len(imgpoints_projected)
            errors.append(error)
            total_error += error
            num_points += 1

        mean_error = total_error / num_points

        if ShowPlot:
            # Plotting the bar graph
            fig, ax = plt.subplots()
            img_indices = range(1, len(errors) + 1)
            ax.bar(img_indices, errors)
            ax.set_xlabel('Image Index')
            ax.set_ylabel('Reprojection Error in Pixels')
            ax.set_title('Reprojection Error for Each Image')
            plt.show()
            print(errors)
            #Save the bar plot as a .png
            fig.savefig(os.path.join(self.ws_root, 'ReprojectionError.png'))
        return mean_error

def test():
    """
    This function is used to test the CameraCalibration class
    :return: None
    """
    root_folder = os.path.dirname(os.path.abspath(__file__))
    image_folder = os.path.join(root_folder, "data/images")
    PoseFolder = os.path.join(root_folder, "data/tcp_poses")
    calib = CameraCalibration(
        image_folder=image_folder,
        Transforms_folder=PoseFolder,
        pattern_size=(9, 7),
        square_size=20/1000,
        ShowCorners=False,
        ShowProjectError=True,
    )
    calib.run()

if __name__== "__main__":
    test()