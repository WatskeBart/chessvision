import cv2
from ultralytics import YOLO

from detect_chessboard import find_board_corners, warp_board
from settings import settings

FILES = "abcdefgh"
RANKS = "87654321"  # rank 8 first, matches a top-left = a8 orientation


def square_name(col, row):
    if settings.flip_orientation:
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
    model = YOLO(str(settings.pieces_model_path))
    cap = cv2.VideoCapture(str(settings.stream_url))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        corners = find_board_corners(frame)
        view = frame

        if corners is not None:
            warped = warp_board(frame, corners)
            view = warped

            results = model(warped, verbose=True)[0]
            view = results.plot(font_size=4, line_width=1)

            board = detections_to_board(results, warped.shape[1], warped.shape[0])
            if board:
                occupied = ", ".join(
                    f"{sq}:{label}" for sq, (label, _) in sorted(board.items())
                )
                print(occupied)

        cv2.imshow("Chess piece detection", scale_for_display(view, settings.display_size))
        if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
