import cv2
import numpy as np
import math

# Replace with your stream URL
stream_url = "http://10.42.0.177:4444/stream"
cap = cv2.VideoCapture(stream_url)

while True:
    ret, frame = cap.read()
    if not ret:
        break


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_a = cv2.GaussianBlur(gray, (3, 3), 0)

    threshold_kernel_size = blurred_a.shape[0] // 8 * 2 + 1
    thresh = cv2.adaptiveThreshold(
        blurred_a, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, threshold_kernel_size, 3
    )

    edges = cv2.Canny(thresh, 50, 200, None, 3)

    # Close gaps in the border
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)


    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # largest quadrilateral
    board = None
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            board = approx.reshape(4, 2)
            break

    if board is not None:
        # Order points in clockwise order
        rect = np.zeros((4, 2), dtype="float32")
        s = board.sum(axis=1)
        rect[0] = board[np.argmin(s)]
        rect[2] = board[np.argmax(s)]

        diff = np.diff(board, axis=1)
        rect[1] = board[np.argmin(diff)]
        rect[3] = board[np.argmax(diff)]

        (tl, tr, br, bl) = rect

        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        max_width = max(int(width_a), int(width_b))

        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        max_height = max(int(height_a), int(height_b))

        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped_a = cv2.warpPerspective(frame, M, (max_width, max_height))

    cv2.imshow('MJPEG Stream', warped_a if board is not None else frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()   
