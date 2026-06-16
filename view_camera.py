import cv2

from settings import settings

cap = cv2.VideoCapture(str(settings.stream_url))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("MJPEG Stream", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
