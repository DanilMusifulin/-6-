import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import cv2
import numpy as np
import yaml
import sys

class ArucoLocatorNode(Node):
    def __init__(self):
        super().__init__('aruco_locator_node')
        
        # Топик для публикации координат робота
        self.pose_pub = self.create_publisher(PoseStamped, 'robot_pose', 10)
        
        # Частота обработки кадров (30 FPS)
        self.timer = self.create_timer(1.0 / 30.0, self.process_frame)
        
        # Физические параметры твоих маркеров
        self.marker_size = 0.270  # 270 мм
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_250)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        # ОБНОВЛЕННАЯ КАРТА ПОД ТВОИ РЕАЛЬНЫЕ ID 1 и 2
        # Центр Маркера №1 — это точка (0,0) на полу
        self.marker_world_map = {
            1: np.array([0.0, 0.0, 2.71]),    # Стартовый маркер
            2: np.array([1.183, 0.0, 2.71])   # Второй маркер на расстоянии 1.183м по X
        }
        
        # Локальные 3D координаты углов маркера для SolvePnP
        self.marker_3d_edges = np.array([
            [-self.marker_size / 2,  self.marker_size / 2, 0],
            [ self.marker_size / 2,  self.marker_size / 2, 0],
            [ self.marker_size / 2, -self.marker_size / 2, 0],
            [-self.marker_size / 2, -self.marker_size / 2, 0]
        ], dtype=np.float32)
        
        # Загрузка параметров калибровки камеры
        self.load_calibration()
        
        # Инициализация веб-камеры Logitech C270
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.get_logger().info("Нода локализации ArUco (ID 1 и 2) запущена!")

    def load_calibration(self):
        try:
            with open("calibration.yaml", "r") as f:
                calib_data = yaml.safe_load(f)
            self.cam_matrix = np.array(calib_data["camera_matrix"])
            self.dist_coeffs = np.array(calib_data["distortion_coefficients"])
            self.get_logger().info("Калибровка успешно загружена.")
        except Exception as e:
            self.get_logger().error(f"Ошибка загрузки калибровки: {e}")
            rclpy.shutdown()

    def rotation_matrix_to_quaternion(self, R):
        """Конвертер матрицы вращения 3х3 в кватернион [x, y, z, w]"""
        t = np.trace(R)
        if t > 0:
            M = np.sqrt(t + 1.0) * 2
            qw = 0.25 * M
            qx = (R[2, 1] - R[1, 2]) / M
            qy = (R[0, 2] - R[2, 0]) / M
            qz = (R[1, 0] - R[0, 1]) / M
        else:
            if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
                M = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                qw = (R[2, 1] - R[1, 2]) / M
                qx = 0.25 * M
                qy = (R[0, 1] + R[1, 0]) / M
                qz = (R[0, 2] + R[2, 0]) / M
            elif R[1, 1] > R[2, 2]:
                M = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                qw = (R[0, 2] - R[2, 0]) / M
                qx = (R[0, 1] + R[1, 0]) / M
                qy = 0.25 * M
                qz = (R[1, 2] + R[2, 1]) / M
            else:
                M = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                qw = (R[1, 0] - R[0, 1]) / M
                qx = (R[0, 2] + R[2, 0]) / M
                qy = (R[1, 2] + R[2, 1]) / M
                qz = 0.25 * M
        return [qx, qy, qz, qw]

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        corners, ids, rejected = self.detector.detectMarkers(frame)

        if ids is not None:
            poses_x, poses_y, poses_z = [], [], []
            final_quat = [0.0, 0.0, 0.0, 1.0]
            detected_any = False

            for i in range(len(ids)):
                marker_id = int(ids[i][0])
                
                if marker_id in self.marker_world_map:
                    marker_2d_corners = corners[i][0].astype(np.float32)
                    success, rvec, tvec = cv2.solvePnP(
                        self.marker_3d_edges, marker_2d_corners, 
                        self.cam_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
                    )

                    if success:
                        # Считаем положение камеры относительно маркера
                        R, _ = cv2.Rodrigues(rvec)
                        R_inv = R.T
                        cam_pos_relative = -np.dot(R_inv, tvec).flatten()

                        # Получаем глобальные координаты текущего маркера из карты
                        g_marker = self.marker_world_map[marker_id]
                        
                        # Расчет глобального положения камеры
                        abs_x = g_marker[0] + cam_pos_relative[0]
                        abs_y = g_marker[1] + cam_pos_relative[1]
                        abs_z = g_marker[2] - np.abs(cam_pos_relative[2]) # Высота от пола

                        poses_x.append(abs_x)
                        poses_y.append(abs_y)
                        poses_z.append(abs_z)
                        
                        if not detected_any:
                            final_quat = self.rotation_matrix_to_quaternion(R_inv)
                            detected_any = True

            # Публикуем усредненные координаты, если задетектировали маркеры 1 или 2
            if detected_any:
                msg = PoseStamped()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = 'map'
                
                msg.pose.position.x = float(np.mean(poses_x))
                msg.pose.position.y = float(np.mean(poses_y))
                msg.pose.position.z = float(np.mean(poses_z))
                
                msg.pose.orientation.x = final_quat[0]
                msg.pose.orientation.y = final_quat[1]
                msg.pose.orientation.z = final_quat[2]
                msg.pose.orientation.w = final_quat[3]
                
                self.pose_pub.publish(msg)
                
                self.get_logger().info(
                    f"Pose -> X: {msg.pose.position.x:.2f}m, Y: {msg.pose.position.y:.2f}m, Z: {msg.pose.position.z:.2f}m"
                )

    def __del__(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()

def main(args=None):
    rclpy.init(args=args)
    node = ArucoLocatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()