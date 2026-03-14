import cv2

# correct video path
video = cv2.VideoCapture("videos/road_video.mp4")

if not video.isOpened():
    print("Error: Could not open video")

while True:
    ret, frame = video.read()

    if not ret:
        break

    cv2.imshow("Self Driving Camera", frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()