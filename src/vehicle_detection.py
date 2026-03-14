import cv2
from ultralytics import YOLO

# load YOLO model
model = YOLO("yolov8n.pt")

video = cv2.VideoCapture("videos/road_video.mp4")

while True:
    ret, frame = video.read()
    if not ret:
        break

    # run YOLO detection
    results = model(frame)

    # draw bounding boxes
    annotated_frame = results[0].plot()

    cv2.imshow("Vehicle Detection", annotated_frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()