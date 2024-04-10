import cv2
from torch.cuda import is_available
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8s.pt')


title = 'YOLOv8 Tracking on cuda' if is_available() else 'YOLOv8 Tracking on cpu'

# Open the video file
video_path = input("Введите ссылку на видеопоток или видео: ")
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame, imgsz=256)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow(title, annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
