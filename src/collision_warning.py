import cv2
from ultralytics import YOLO

# load YOLO model
model = YOLO("yolov8n.pt")

video = cv2.VideoCapture("videos/road_video.mp4")

while True:
    ret, frame = video.read()
    if not ret:
        break

    results = model(frame)

    for result in results:
        boxes = result.boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            label = model.names[cls]

            height = y2 - y1

            # draw bounding box
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,f"{label} {conf:.2f}",(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

            # collision warning condition
            if height > 80 and label in ["car","truck","bus","motorcycle"]:
                cv2.putText(frame,
                            "COLLISION WARNING!",
                            (50,50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0,0,255),
                            3)

    cv2.imshow("Collision Warning System", frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()