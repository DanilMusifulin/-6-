import cv2
import numpy as np
import yaml

# Настройки шахматной доски (количество ВНУТРЕННИХ углов)
# Если доска 10х7 квадратов, то внутренних стыков будет 9х6
CHECKERBOARD = (9, 6)

# Физический размер стороны одного квадрата в метрах (например, 2.5 см = 0.025)
# Измерь линейкой на своем А4!
SQUARE_SIZE = 0.023 

# Критерии для субпиксельной точности поиска углов
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Массивы для хранения 3D и 2D точек
objpoints = [] # 3D точки в реальном мире
imgpoints = [] # 2D точки на картинке

# Подготовка 3D координат (0,0,0), (1,0,0), (2,0,0) ...
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Инструкция:")
print("1. Показывай шахматную доску камере под разными углами.")
print("2. Нажимай 'Space' (Пробел), чтобы сохранить кадр.")
print("3. Нужно собрать 15-20 успешных кадров.")
print("4. Нажми 'q' для выхода и подсчета калибровки.")

saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Быстрый поиск углов для отображения в реальном времени
    ret_corners, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    
    frame_vis = frame.copy()
    if ret_corners:
        # Рисуем сетку, если нашли
        cv2.drawChessboardCorners(frame_vis, CHECKERBOARD, corners, ret_corners)
        
    cv2.putText(frame_vis, f"Saved frames: {saved_count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Calibration', frame_vis)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '): # Пробел
        if ret_corners:
            objpoints.append(objp)
            # Уточняем координаты углов
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            saved_count += 1
            print(f"Кадр {saved_count} сохранен!")
        else:
            print("Углы не найдены, выровняй доску!")
            
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if saved_count > 10:
    print("Вычисляю матрицу калибровки...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    # Сохраняем в YAML для ROS 2 и OpenCV
    data = {
        "camera_matrix": mtx.tolist(),
        "distortion_coefficients": dist.tolist()
    }
    
    with open("calibration.yaml", "w") as f:
        yaml.dump(data, f)
        
    print("Калибровка успешно сохранена в файл 'calibration.yaml'!")
    print("\nМатрица камеры:\n", mtx)
    print("\nДисторсия:\n", dist)
else:
    print("Слишком мало кадров для калибровки. Нужно хотя бы 10.")