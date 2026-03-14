import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort

# Load YOLO model
model = YOLO("yolov8n.pt")

# Initialize tracker
tracker = Sort()

# Store previous positions for intrusion detection
previous_positions = {}

# Open video
video = cv2.VideoCapture("videos/speedbreaker_video.mp4")


# ---------- REGION OF INTEREST ----------
def region_of_interest(image):
    height, width = image.shape

    polygons = np.array([
        [
            (0, int(height*0.6)),
            (width, int(height*0.6)),
            (width, int(height*0.3)),
            (0, int(height*0.3))
        ]
    ], np.int32)

    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)

    return cv2.bitwise_and(image, mask)


# ---------- SPEED BREAKER DETECTION ----------
def detect_speedbreaker(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray,(5,5),0)

    edges = cv2.Canny(blur,50,150)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi/180,
        80,
        minLineLength=120,
        maxLineGap=20
    )

    if lines is not None:

        horizontal_count = 0

        for line in lines:
            x1,y1,x2,y2 = line[0]

            if abs(y2-y1) < 8:
                horizontal_count += 1

        if horizontal_count > 3:

            cv2.putText(frame,
                        "SPEED BREAKER AHEAD",
                        (50,150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,255,255),
                        3)


while True:

    ret, frame = video.read()

    if not ret:
        break

    frame_height, frame_width = frame.shape[:2]

    # ---------- LANE DETECTION ----------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)

    edges = cv2.Canny(blur,50,150)

    cropped_edges = region_of_interest(edges)

    lines = cv2.HoughLinesP(
        cropped_edges,
        1,
        np.pi/180,
        50,
        minLineLength=40,
        maxLineGap=5
    )

    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),3)

    # ---------- SPEED BREAKER ----------
    detect_speedbreaker(frame)

    # ---------- OBJECT DETECTION ----------
    results = model(frame)

    detections = []
    labels = []

    for result in results:

        boxes = result.boxes

        for box in boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            label = model.names[cls]

            detections.append([x1, y1, x2, y2, conf])
            labels.append(label)

    if len(detections) > 0:
        detections = np.array(detections)
    else:
        detections = np.empty((0,5))

    # ---------- TRACKING ----------
    tracks = tracker.update(detections)

    for i, track in enumerate(tracks):

        x1, y1, x2, y2, track_id = map(int, track)

        height = y2 - y1
        center_x = (x1 + x2) // 2

        if i < len(labels):
            object_name = labels[i]
        else:
            object_name = "Vehicle"

        # ---------- RISK LEVEL ----------
        if height > 90:
            risk = "DANGER"
            color = (0,0,255)

        elif height > 60:
            risk = "CLOSE"
            color = (0,165,255)

        else:
            risk = "SAFE"
            color = (0,255,0)

        # Draw box
        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)

        text = f"{object_name} ID {track_id} - {risk}"

        cv2.putText(frame,
                    text,
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2)

        # ---------- SUDDEN INTRUSION ----------
        lane_center = frame_width // 2

        if track_id in previous_positions:

            prev_x = previous_positions[track_id]

            lateral_movement = abs(center_x - prev_x)

            if lateral_movement > 30 and abs(center_x - lane_center) < frame_width * 0.3:

                cv2.putText(frame,
                            "SUDDEN INTRUSION!",
                            (50,100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0,0,255),
                            3)

        previous_positions[track_id] = center_x


    cv2.imshow("Autonomous Driving System", frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break


video.release()
cv2.destroyAllWindows()