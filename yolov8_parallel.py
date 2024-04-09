import cv2
import threading
from time import time
from queue import Queue
from ultralytics import YOLO
from torch.cuda import is_available

# from opencv_draw_annotation import draw_bounding_box
title = 'YOLOv8 Tracking on cuda' if is_available() else 'YOLOv8 Tracking on cpu'
# Display the annotated frame

# Загрузка моделей YOLOv8
car_model = YOLO('yolov8s.pt')
sign_model = YOLO('best(s).pt')


cams = {'Доватора - Блюхера': "https://cdn.cams.is74.ru/hls/playlists/multivariant.m3u8?uuid=8e6a6c46-9f40-4f72-8c8c"
                              "-305ed67effdc",
        'Артиллерийская - 1й Пятилетки': "https://cdn.cams.is74.ru/hls/playlists/multivariant.m3u8?uuid=16ee1359-be80"
                                         "-461d-a277-26e2f8c0ab03",
        'Ленина - Артиллерийская': "https://cdn.cams.is74.ru/hls/playlists/multivariant.m3u8?uuid=04bb24b0-dee6-4848"
                                   "-8963-163ab3bcc25c"}

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
cv2.resizeWindow(title, 1600, 900)


def process_frame_car_model(fr):
    boxes = car_model.predict(fr, imgsz=256, iou=0.5, conf=0.45, stream_buffer=True, verbose=False)
    car_queue.put(boxes)


def process_frame_sign_model(fr):
    boxes = sign_model.predict(fr, imgsz=256, iou=0.5, conf=0.45, stream_buffer=True, verbose=False)
    sign_queue.put(boxes)


while True:
    # Получение кадра из видеопотока
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (int(frame.shape[1] * 0.5), int(frame.shape[0] * 0.5)))
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

        frame = car_result.plot()
        frame = sign_result.plot(img=frame)

        # # Отрисовка bbox на кадре
        # for box, cls, conf in zip(car_result.boxes.xyxy, car_result.boxes.cls, car_result.boxes.conf):
        #     x1, y1, x2, y2 = [int(c) for c in box]
        #     # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #     draw_bounding_box(frame, (x1, y1, x2, y2), labels=[car_result.names[cls.item()]+' '+str(round(conf.item(), 2))],
        #                       color='green', )
        #
        # # Отрисовка bbox на кадре
        # for box, cls, conf in zip(sign_result.boxes.xyxy, sign_result.boxes.cls, sign_result.boxes.conf):
        #     x1, y1, x2, y2 = [int(c) for c in box]
        #     # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #     draw_bounding_box(frame, (x1, y1, x2, y2), labels=[sign_result.names[cls.item()]+' '+str(round(conf.item(), 2))],
        #                       color='red')

        end_time = time()
        fps = 1 // (end_time - start_time)
        cv2.putText(frame, f'{str(fps)}', (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f'{"{:.2f} ms".format(avg_inf)}', (frame.shape[1] - 105, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow(title, frame)

        # Выход по нажатию 'q'
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) & 0xFF == ord('й'):
            break
    else:
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
