import argparse
import sys
from datetime import datetime

import cv2
from ultralytics import YOLO

from detect_corners_cv import board_quad_from_corners, find_corners, warp_board
from settings import settings
from track_game import GameTracker, normalize_fen

FILES = "abcdefgh"
RANKS = "87654321"  # rank 8 first, matches a top-left = a8 orientation

TOGGLE_HELP_LINES = [
    "Keyboard toggles:",
    "  h  show/hide this help",
    "  d  toggle detection overlay (boxes/labels)",
    "  c  toggle corner overlay on raw frame",
    "  k  lock/unlock board transform (calibrate once)",
    "  p  toggle printing board state to stdout",
    "  f  toggle board flip orientation",
    "  r  start/stop recording the game (PGN + FEN log)",
    "  s  save current frame to snapshot_<n>.png",
    "  ESC  quit",
]


def square_name(col, row):
    if settings.flip_orientation:
        col, row = 7 - col, 7 - row
    return f"{FILES[col]}{RANKS[row]}"


def detections_to_board(result, board_w, board_h, padding=0):
    """Map each detected piece to a square based on the box's bottom-center
    point (a piece's base sits on its square, the box center does not)."""
    board = {}
    cell_w = (board_w - 2 * padding) / 8
    cell_h = (board_h - 2 * padding) / 8

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx = (x1 + x2) / 2 - padding
        cy = y2 - padding  # bottom edge of the box, offset by padding

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
    """Draw detected corner points and the board quad outline on a copy of frame.

    Either may be None: when the transform is locked there are no live corner
    points, only the cached quad to outline."""
    out = frame.copy()
    if corners is not None:
        for pt in corners:
            x, y = int(pt[0][0]), int(pt[0][1])
            cv2.circle(out, (x, y), 6, (0, 255, 0), -1)
    if quad is not None:
        pts = quad.reshape((-1, 1, 2)).astype(int)
        cv2.polylines(out, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
    return out


def draw_lock_indicator(frame):
    """Draw a 'BOARD LOCKED' marker (top-right) while the transform is locked."""
    out = frame.copy()
    w = out.shape[1]
    cv2.putText(
        out, "BOARD LOCKED", (w - 210, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        (0, 255, 255), 2, cv2.LINE_AA,
    )
    return out


def draw_record_overlay(frame, tracker):
    """Draw a red REC indicator with move count and last move, bottom-left."""
    out = frame.copy()
    h = out.shape[0]
    text = f"REC  moves: {tracker.move_count}"
    if tracker.last_san:
        text += f"  last: {tracker.last_san}"
    cv2.circle(out, (18, h - 16), 7, (0, 0, 255), -1)
    cv2.putText(
        out, text, (34, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        (0, 0, 255), 2, cv2.LINE_AA,
    )
    return out


def start_recording(start_fen=None):
    """Create a GameTracker writing to timestamped files in settings.games_dir."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pgn_path = settings.games_dir / f"game_{ts}.pgn"
    fen_path = settings.games_dir / f"game_{ts}.fen.log"
    tracker = GameTracker(
        pgn_path,
        fen_path,
        start_fen=start_fen,
        stability_frames=settings.game_stability_frames,
        debug=settings.game_debug,
    )
    origin = "custom position" if start_fen else "standard opening"
    print(f"[rec] recording started from {origin} -> {pgn_path}")
    return tracker


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
    gui_status = "ok" if has_gui else "not detected in build info (may still work)"
    print(f"[display] DISPLAY={display!r}  OpenCV GUI: {gui_status}")


def main(start_fen=None):
    _check_display()
    print("\n".join(TOGGLE_HELP_LINES))
    if start_fen:
        print(f"[init] recording will start from: {start_fen}")

    model_path = settings.pieces_model_path
    print(f"[init] loading model: {model_path}", flush=True)
    if not model_path.exists():
        print(f"ERROR: model not found: {model_path.resolve()}", flush=True)
        raise SystemExit(1)
    model = YOLO(str(model_path), task="detect")
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
    tracker = None  # GameTracker while recording, else None
    board_visible = True  # tracks corner-detection state for recording logs
    locked_quad = None  # cached board quad once calibrated, else detect per frame

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

        # Locate the board. Once calibrated, reuse the locked quad and skip
        # detection entirely (so pieces occluding the grid can't break tracking);
        # otherwise detect the corners fresh each frame.
        if locked_quad is not None:
            corners = None
            quad = locked_quad
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners = find_corners(gray)
            quad = board_quad_from_corners(corners) if corners is not None else None

        view = frame

        # While recording without a lock, surface when the board can't be located
        # (the corner detector often fails once pieces occlude the inner grid),
        # which silently pauses tracking. A locked transform never drops out.
        if (
            tracker is not None
            and locked_quad is None
            and (quad is not None) != board_visible
        ):
            board_visible = quad is not None
            if board_visible:
                print("[rec] board reacquired — tracking resumed")
            else:
                print(
                    "[rec] board NOT detected (corners hidden by pieces?) — tracking "
                    "paused. Tip: press 'k' while the board is visible to lock it."
                )

        if quad is not None:
            warped = warp_board(frame, quad, padding=settings.warp_padding)

            results = model(
                warped, verbose=False, conf=settings.pieces_conf_threshold
            )[0]

            if show_detections:
                view = results.plot(font_size=4, line_width=1)
            else:
                view = warped

            board = detections_to_board(
                results, warped.shape[1], warped.shape[0], padding=settings.warp_padding
            )
            if print_board and board:
                occupied = ", ".join(
                    f"{sq}:{label}" for sq, (label, _) in sorted(board.items())
                )
                print(occupied)

            if tracker is not None:
                tracker.update(board)

        # The corner/quad overlay always belongs on the raw frame (matching
        # detect_corners_cv.py). corners is None when the transform is locked, so
        # only the quad outline shows then.
        if show_corners:
            view = draw_corners_overlay(frame, corners, quad)

        if locked_quad is not None:
            view = draw_lock_indicator(view)

        if tracker is not None:
            view = draw_record_overlay(view, tracker)

        if show_help:
            view = draw_help_overlay(view)

        try:
            cv2.imshow("Chess piece detection", scale_for_display(view, settings.display_size))
        except Exception as e:
            print(f"[display] imshow failed: {type(e).__name__}: {e}", flush=True)
            break

        key = cv2.waitKey(1) & 0xFF
        if frame_count == 1:
            print(f"[loop] waitKey returned {key}", flush=True)
        if key == 27:  # ESC — quit
            if tracker is not None:
                tracker.finalize()
                print(f"[rec] recording stopped -> {tracker.pgn_path}")
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
        elif key == ord("k"):
            if locked_quad is None:
                if quad is not None:
                    locked_quad = quad.copy()
                    board_visible = True
                    print(
                        "[calib] board transform LOCKED — reusing it every frame; "
                        "corner detection is now skipped. Press 'k' to unlock."
                    )
                else:
                    print(
                        "[calib] cannot lock: no board detected this frame. "
                        "Make the board fully visible, then press 'k'."
                    )
            else:
                locked_quad = None
                print("[calib] board transform unlocked — detecting per frame again.")
        elif key == ord("p"):
            print_board = not print_board
            print(f"[toggle] print board state: {'on' if print_board else 'off'}")
        elif key == ord("f"):
            flip = not flip
            settings.flip_orientation = flip
            corner = "top-left" if flip else "bottom-right"
            print(
                f"[toggle] flip orientation: {'on' if flip else 'off'} "
                f"(A1 at {corner})"
            )
        elif key == ord("r"):
            if tracker is None:
                tracker = start_recording(start_fen)
            else:
                tracker.finalize()
                print(
                    f"[rec] recording stopped ({tracker.move_count} moves) "
                    f"-> {tracker.pgn_path}"
                )
                tracker = None
        elif key == ord("s"):
            path = f"snapshot_{snapshot_count}.png"
            cv2.imwrite(path, view)
            print(f"[snapshot] saved {path}")
            snapshot_count += 1

    cap.release()
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect a live chess board and pieces, and optionally record "
        "the game (press 'r' to start/stop)."
    )
    parser.add_argument(
        "--from-fen",
        metavar="FEN",
        help="Record starting from this position instead of the standard "
        "opening. Accepts a full FEN, or just the piece-placement field "
        "(in which case White is assumed to move first).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    start_fen = None
    if args.from_fen:
        try:
            start_fen = normalize_fen(args.from_fen)
        except ValueError as e:
            print(f"ERROR: invalid --from-fen: {e}", file=sys.stderr)
            raise SystemExit(2) from None
    main(start_fen)
