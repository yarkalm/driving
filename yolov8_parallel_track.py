import threading
from collections import defaultdict
from queue import Queue
from time import time
from lane_detect2 import process_image
import cv2
import numpy as np
from torch import cat
from torch.cuda import is_available
from ultralytics import YOLO
from ultralytics.utils.plotting import colors

from lane_detect import lane_finding, perspective_transformer

title = 'YOLOv8 Tracking on cuda' if is_available() else 'YOLOv8 Tracking on cpu'

perspective_transformer()

# Загрузка моделей YOLOv8
car_model = YOLO('yolov8s.pt')
sign_model = YOLO('roadsings_weights.pt')

cams = {'Доватора - Блюхера': "https://cdn.cams.is74.ru/hls/playlists/multivariant.m3u8?uuid=8e6a6c46-9f40-4f72-8c8c"
                              "-305ed67effdc",
        'Артиллерийская - 1й Пятилетки': "https://cdn.cams.is74.ru/hls/playlists/multivariant.m3u8?uuid=16ee1359-be80"
                                         "-461d-a277-26e2f8c0ab03",
        'Ленина - Артиллерийская': "https://cdn.cams.is74.ru/hls/playlists/multivariant.m3u8?uuid=04bb24b0-dee6-4848"
                                   "-8963-163ab3bcc25c",
        'Yar': "https://cam15.yar-net.ru/OL6y9ghd/mono.m3u8?token=7bc9b4f6796ede3ab681067a74a95131",
        'Car city driver': "3d_car_driving_test.mp4",
        'Web-Cam': 0}

cams_id = {i: item for i, item in enumerate(cams.keys())}

print("Список камер 'Интерсвязь' Челябинск")
for cam_id, cam in cams_id.items():
    print("{0}: {1}".format(cam_id, cams_id[cam_id]))

# Открытие видеопотока
cap = cv2.VideoCapture(cams[cams_id[int(input("Выберите номер камеры из списка: "))]])

# Очереди для хранения результатов
car_queue = Queue()
sign_queue = Queue()

cv2.namedWindow(title, cv2.WINDOW_NORMAL)
cv2.resizeWindow(title, 1280, 960)

track_history = defaultdict(lambda: [])


def process_frame_car_model(fr):
    boxes = car_model.track(fr, imgsz=256, iou=0.5, conf=0.45, stream_buffer=True, verbose=False, persist=True)
    car_queue.put(boxes)


def process_frame_sign_model(fr):
    boxes = sign_model.track(fr, imgsz=256, iou=0.5, conf=0.20, stream_buffer=True, verbose=False, persist=True)
    sign_queue.put(boxes)


while True:
    # Получение кадра из видеопотока
    ret, frame = cap.read()
    if ret:
        # frame = cv2.resize(frame, (int(frame.shape[1] * 0.5), int(frame.shape[0] * 0.5)))
        frame = cv2.resize(frame, (1280, 720))
        start_time = time()

        # Параллельная обработка кадра двумя моделями
        thread1 = threading.Thread(target=process_frame_car_model, args=(frame,))
        thread2 = threading.Thread(target=process_frame_sign_model, args=(frame,))

        thread1.start()
        thread2.start()

        # Получение результатов из очередей

        car_result = car_queue.get()[0]
        sign_result = sign_queue.get()[0]
        avg_inf = round((car_result.speed['inference'] + sign_result.speed['inference']) / 2, 2)

        frame = sign_result.plot()
        frame = car_result.plot(img=frame)

        cv2.putText(frame, f'{"{:.2f} ms".format(avg_inf)}', (frame.shape[1] - 105, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        # if sign_result.boxes.id is not None and car_result.boxes.id is not None:
        #     boxes = cat((car_result.boxes.xywh.cpu(), sign_result.boxes.xywh.cpu()), dim=0)
        #     track_ids = car_result.boxes.id.int().cpu().tolist() + sign_result.boxes.id.int().cpu().tolist()
        #     clss = car_result.boxes.cls.cpu() + sign_result.boxes.cls.cpu()
        # elif car_result.boxes.id is not None:
        #     track_ids = car_result.boxes.id.int().cpu().tolist()
        #     boxes = car_result.boxes.xywh.cpu()
        #     clss = car_result.boxes.cls.cpu()
        # elif sign_result.boxes.id is not None:
        #     track_ids = sign_result.boxes.id.int().cpu().tolist()
        #     boxes = sign_result.boxes.xywh.cpu()
        #     clss = sign_result.boxes.cls.cpu()
        # else:
        #     track_ids = []
        #     boxes = []
        #     clss = []

        # # Plot the tracks
        # for box, track_id, cls in zip(boxes, track_ids, clss):
        #     x, y, w, h = box
        #     track = track_history[track_id]
        #     track.append((float(x), float(y)))  # x, y center point
        #     if len(track) > 30:  # retain 90 tracks for 90 frames
        #         track.pop(0)
        #
        #     # Draw the tracking lines
        #     points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        #     cv2.polylines(frame, [points], isClosed=False, color=colors(cls, bgr=True), thickness=5,
        #                   lineType=cv2.LINE_4, shift=0)
        frame = lane_finding(frame)
        end_time = time()
        fps = 1 // (end_time - start_time)
        cv2.putText(frame, f'{str(fps)}', (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(title, frame)

        # Выход по нажатию 'q'
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) & 0xFF == ord('й'):
            break
    else:
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
