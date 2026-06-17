import cv2
from ultralytics import YOLO

from detect_corners_cv import board_quad_from_corners, find_corners, warp_board
from settings import settings

FILES = "abcdefgh"
RANKS = "87654321"  # rank 8 first, matches a top-left = a8 orientation

TOGGLE_HELP_LINES = [
    "Keyboard toggles:",
    "  h  show/hide this help",
    "  d  toggle detection overlay (boxes/labels)",
    "  c  toggle corner overlay on raw frame",
    "  p  toggle printing board state to stdout",
    "  f  toggle board flip orientation",
    "  s  save current frame to snapshot_<n>.png",
    "  ESC  quit",
]


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


def draw_help_overlay(frame):
    """Render the help menu as a semi-transparent box in the top-left corner."""
    out = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness, pad = 0.55, 1, 10
    line_h = 22
    box_w = 480
    box_h = pad * 2 + line_h * len(TOGGLE_HELP_LINES)
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (box_w, box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, out, 0.4, 0, out)
    for i, line in enumerate(TOGGLE_HELP_LINES):
        y = pad + line_h * i + line_h - 4
        cv2.putText(
            out, line, (pad, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA
        )
    return out


def draw_corners_overlay(frame, corners, quad):
    """Draw detected corner points and board quad outline onto a copy of frame."""
    out = frame.copy()
    for pt in corners:
        x, y = int(pt[0][0]), int(pt[0][1])
        cv2.circle(out, (x, y), 6, (0, 255, 0), -1)
    pts = quad.reshape((-1, 1, 2)).astype(int)
    cv2.polylines(out, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
    return out


def _check_display():
    import os
    import sys

    display = os.environ.get("DISPLAY", "")
    if not display:
        print(
            "ERROR: $DISPLAY is not set. Run with X11 forwarding (ssh -X) "
            "and ensure DISPLAY is exported, e.g.:\n"
            "  DISPLAY=:10.0 uv run detect_pieces.py",
            file=sys.stderr,
        )
        sys.exit(1)

    # Check that OpenCV was built with a GUI backend (GTK or Qt).
    build_info = cv2.getBuildInformation()
    has_gui = any(
        f"{lib}:" in build_info and "YES" in build_info.split(f"{lib}:")[1].split("\n")[0]
        for lib in ("GTK+", "GTK", "Qt5", "Qt6", "QT")
    )
    if not has_gui:
        print(
            "ERROR: OpenCV has no GUI backend (no GTK/Qt). "
            "Install system OpenCV instead:\n"
            "  sudo apt-get install python3-opencv\n"
            "and remove 'opencv-python' from pyproject.toml.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[display] DISPLAY={display!r}  OpenCV GUI: ok")


def main():
    _check_display()
    #print("\n".join(TOGGLE_HELP_LINES))

    model_path = settings.pieces_model_path
    print(f"[init] loading model: {model_path}", flush=True)
    if not model_path.exists():
        print(f"ERROR: model not found: {model_path.resolve()}", flush=True)
        raise SystemExit(1)
    model = YOLO(str(model_path))
    print("[init] model loaded", flush=True)

    url = str(settings.stream_url)
    print(f"[init] opening stream: {url}", flush=True)
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print(f"ERROR: cannot open stream: {url}", flush=True)
        raise SystemExit(1)
    print("[init] stream opened", flush=True)

    show_detections = True
    show_corners = False
    show_help = False
    print_board = False
    flip = settings.flip_orientation
    snapshot_count = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[stream] read failed, reconnecting...", flush=True)
            cap.release()
            cap = cv2.VideoCapture(url)
            continue

        frame_count += 1
        if frame_count == 1:
            print("[loop] first frame received", flush=True)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = find_corners(gray)
        view = frame

        if corners is not None:
            quad = board_quad_from_corners(corners)
            warped = warp_board(frame, quad)

            results = model(
                warped, verbose=False, conf=settings.pieces_conf_threshold
            )[0]

            if show_detections:
                view = results.plot(font_size=4, line_width=1)
            else:
                view = warped

            if show_corners:
                view = draw_corners_overlay(view, corners, quad)

            board = detections_to_board(results, warped.shape[1], warped.shape[0])
            if print_board and board:
                occupied = ", ".join(
                    f"{sq}:{label}" for sq, (label, _) in sorted(board.items())
                )
                print(occupied)
        elif show_corners:
            # No board found — show raw frame so the user can see what's happening
            view = frame

        if show_help:
            view = draw_help_overlay(view)

        try:
            cv2.imshow("Chess piece detection", scale_for_display(view, settings.display_size))
        except cv2.error as e:
            print(f"[display] imshow failed: {e}", flush=True)
            break

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC — quit
            cap.release()
            break
        elif key == ord("h"):
            show_help = not show_help
        elif key == ord("d"):
            show_detections = not show_detections
            print(f"[toggle] detection overlay: {'on' if show_detections else 'off'}")
        elif key == ord("c"):
            show_corners = not show_corners
            print(f"[toggle] corner overlay: {'on' if show_corners else 'off'}")
        elif key == ord("p"):
            print_board = not print_board
            print(f"[toggle] print board state: {'on' if print_board else 'off'}")
        elif key == ord("f"):
            flip = not flip
            settings.flip_orientation = flip
            print(f"[toggle] flip orientation: {'on' if flip else 'off'}")
        elif key == ord("s"):
            path = f"snapshot_{snapshot_count}.png"
            cv2.imwrite(path, view)
            print(f"[snapshot] saved {path}")
            snapshot_count += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
