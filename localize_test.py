import cv2
import numpy as np
import yaml
import sys

# 1. Загрузка калибровки
try:
    with open("calibration.yaml", "r") as f:
        calib_data = yaml.safe_load(f)
    cam_matrix = np.array(calib_data["camera_matrix"])
    dist_coeffs = np.array(calib_data["distortion_coefficients"])
except Exception as e:
    print(f"Ошибка загрузки калибровки: {e}")
    sys.exit()

# 2. Параметры маркера
MARKER_SIZE = 0.270  # 270 мм в метрах
# Словарь твоего маркера (если не определится, попробуй DICT_7X7_50 или DICT_7X7_1000)
ARUCO_DICT = cv2.aruco.DICT_7X7_250 

# Определение 3D-координат углов маркера в его собственной локальной системе (центр в 0,0,0)
# Точки задаются по часовой стрелке, начиная с левого верхнего угла
marker_3d_edges = np.array([
    [-MARKER_SIZE / 2,  MARKER_SIZE / 2, 0],
    [ MARKER_SIZE / 2,  MARKER_SIZE / 2, 0],
    [ MARKER_SIZE / 2, -MARKER_SIZE / 2, 0],
    [-MARKER_SIZE / 2, -MARKER_SIZE / 2, 0]
], dtype=np.float32)

# 3. Инициализация камеры и детектора
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

print("Запуск локализации. Покажи маркер камере. Нажми 'q' для выхода.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    corners, ids, rejected = detector.detectMarkers(frame)

    if ids is not None:
        # Рисуем контур маркера
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        for i in range(len(ids)):
            # Берем 2D координаты углов конкретного маркера
            marker_2d_corners = corners[i][0].astype(np.float32)

            # Решаем задачу PnP (Perspective-n-Point)
            success, rvec, tvec = cv2.solvePnP(
                marker_3d_edges, 
                marker_2d_corners, 
                cam_matrix, 
                dist_coeffs, 
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if success:
                # Отрисовка 3D осей на маркере для визуальной проверки
                cv2.drawFrameAxes(frame, cam_matrix, dist_coeffs, rvec, tvec, 0.1)

                # Магия инверсии: переходим от "маркер относительно камеры" к "камера относительно маркера"
                # Превращаем вектор вращения в матрицу 3х3
                R, _ = cv2.Rodrigues(rvec)
                
                # Инвертируем матрицу поворота (для ортогональной матрицы это просто транспонирование)
                R_inv = R.T
                # Вычисляем позицию камеры: Камера = -R_inv * tvec
                camera_position = -np.dot(R_inv, tvec)

                # Извлекаем координаты (в метрах)
                x_cam, y_cam, z_cam = camera_position.flatten()

                # Выводим координаты прямо на экран
                text = f"Cam Pose: X={x_cam:.2f} Y={y_cam:.2f} Z={z_cam:.2f}"
                cv2.putText(frame, text, (10, 50 + i*30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                print(f"ID: {ids[i][0]} -> X: {x_cam:6.2f}m, Y: {y_cam:6.2f}m, Z: {z_cam:6.2f}m")

    cv2.imshow("Localization Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()