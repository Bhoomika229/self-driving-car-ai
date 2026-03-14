import cv2
import numpy as np

def region_of_interest(image):
    height, width = image.shape

    polygons = np.array([
        [
            (0, height*0.6),
            (width, height*0.6),
            (width, height*0.3),
            (0, height*0.3)
        ]
    ], np.int32)

    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)

    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


video = cv2.VideoCapture("videos/road_video.mp4")

while True:
    ret, frame = video.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    edges = cv2.Canny(blur, 50, 150)

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

    cv2.imshow("Lane Detection", frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()