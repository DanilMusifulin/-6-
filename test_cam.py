import cv2
import sys

# Инициализируем камеру
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Ошибка: Не удалось открыть камеру /dev/video0")
    sys.exit()

# Настройки детектора ArUco
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

print("Нажми 'q' в окне видео, чтобы выйти.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Ошибка: Не удалось получить кадр.")
        break

    # Поиск маркеров на кадре
    corners, ids, rejected = detector.detectMarkers(frame)

    # Если нашли маркеры, подсвечиваем их
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        print(f"Обнаружены маркеры с ID: {ids.flatten()}")

    # Показываем кадр
    cv2.imshow("ArUco Test", frame)

    # Выход по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()