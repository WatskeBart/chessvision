import cv2
import numpy as np
from ultralytics import YOLO

# Replace with your stream URL
stream_url = "http://10.42.0.177:4444/stream"

# Path to a YOLO model fine-tuned on chess pieces (12 classes: 6 piece
# types x {white, black}). yolo26n.pt is only the COCO-pretrained base
# checkpoint and will NOT recognize pieces correctly until fine-tuned -
# see main.py / README for training notes.
#MODEL_PATH = "yolo26n.pt"
MODEL_PATH = "runs/detect/train-2/weights/best.pt"
# If the warped board's top-left corner corresponds to square a8 from the
# camera's point of view, leave this False. Flip to True if your camera/
# board orientation puts a1 in the top-left instead.
FLIP_ORIENTATION = False

# The warped board is only as big as the board appears in the raw camera
# frame, which can be quite small. Scale the displayed view up to at least
# this many pixels on a side so detected pieces/boxes are easy to see.
DISPLAY_SIZE = 800

FILES = "abcdefgh"
RANKS = "87654321"  # rank 8 first, matches a top-left = a8 orientation


def find_board_corners(frame):
    """Locate the board's outer quadrilateral in the raw camera frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    threshold_kernel_size = blurred.shape[0] // 8 * 2 + 1
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, threshold_kernel_size, 3
    )

    edges = cv2.Canny(thresh, 50, 200, None, 3)

    # Close gaps in the border
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)
    return None


def warp_board(frame, corners):
    """Perspective-warp the frame so the board fills a square top-down view."""
    rect = np.zeros((4, 2), dtype="float32")
    s = corners.sum(axis=1)
    rect[0] = corners[np.argmin(s)]
    rect[2] = corners[np.argmax(s)]

    diff = np.diff(corners, axis=1)
    rect[1] = corners[np.argmin(diff)]
    rect[3] = corners[np.argmax(diff)]

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
    warped = cv2.warpPerspective(frame, M, (max_width, max_height))
    return warped


def square_name(col, row):
    if FLIP_ORIENTATION:
        col, row = 7 - col, 7 - row
    return f"{FILES[col]}{RANKS[row]}"


def detections_to_board(result, board_w, board_h):
    """Map each detected piece to a square based on the box's bottom-center
    point (a piece's base sits on its square, the box center does not)."""
    board = {}
    cell_w = board_w / 8
    cell_h = board_h / 8

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx = (x1 + x2) / 2
        cy = y2  # bottom edge of the box

        col = min(7, max(0, int(cx // cell_w)))
        row = min(7, max(0, int(cy // cell_h)))

        label = result.names[int(box.cls[0])]
        conf = float(box.conf[0])

        sq = square_name(col, row)
        # Keep the highest-confidence detection per square
        if sq not in board or conf > board[sq][1]:
            board[sq] = (label, conf)

    return board


def scale_for_display(image, target_size):
    """Upscale (never downscale) a square-ish image so it's easy to view,
    preserving aspect ratio."""
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    if scale <= 1:
        return image
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)


def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(stream_url)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        corners = find_board_corners(frame)
        view = frame

        if corners is not None:
            warped = warp_board(frame, corners)
            view = warped

            results = model(warped, verbose=False)[0]
            view = results.plot()

            board = detections_to_board(results, warped.shape[1], warped.shape[0])
            if board:
                occupied = ", ".join(f"{sq}:{label}" for sq, (label, _) in sorted(board.items()))
                print(occupied)

        cv2.imshow("Chess piece detection", scale_for_display(view, DISPLAY_SIZE))
        if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
