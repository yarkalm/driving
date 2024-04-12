import cv2
import numpy as np
import os
import pickle

# Путь к папке с изображениями
image_dir = "camera_cal"

# Список изображений
images = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)]

# Количество внутренних углов шахматной доски (в количестве клеток)
pattern_size = (8, 6)  # 8x6 внутренних углов

# Шаблон для обнаружения углов
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Массивы для хранения точек на изображениях и реальных мировых координат
objpoints = []  # 3D точки в реальном мире
imgpoints = []  # 2D точки на изображении

# Генерация массива реальных мировых координат шахматной доски
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Попытка найти углы на изображении
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    # Если углы найдены, добавьте их в массив и реальные мировые координаты в другой массив
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

# Выполнение калибрации камеры
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Сохранение результатов калибрации в файл формата pickle
calibration_data = {"mtx": mtx, "dist": dist}
with open("pickle/calibration_data.p", "wb") as f:
    pickle.dump(calibration_data, f)

print("Calibration completed and data saved to 'pickle/calibration_data.p'")
