import cv2

# Replace with your stream URL
stream_url = "http://10.42.0.177:4444/stream"
cap = cv2.VideoCapture(stream_url)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('MJPEG Stream', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()   
