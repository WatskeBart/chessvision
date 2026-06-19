import argparse
import sys
from collections import deque
from datetime import datetime

import chess
import cv2
import numpy as np
from ultralytics import YOLO

from chessvision.core.board import board_quad_from_corners, find_corners, warp_board
from chessvision.core.display import scale_for_display
from chessvision.core.occupancy import changed_squares, square_change_scores
from chessvision.core.tracking import GameTracker, label_to_symbol, normalize_fen
from chessvision.settings import settings

FILES = "abcdefgh"
RANKS = "87654321"  # rank 8 first, matches a top-left = a8 orientation


def rotate_cell(col, row, degrees, n=8):
    """Rotate a cell index within an n×n grid clockwise by 0/90/180/270°.

    Matches how rotating the image clockwise would move that cell, but works on
    indices so the orientation can be corrected in the square mapping while the
    displayed board is left in the raw camera-feed orientation."""
    d = degrees % 360
    if d == 90:
        return n - 1 - row, col
    if d == 180:
        return n - 1 - col, n - 1 - row
    if d == 270:
        return row, n - 1 - col
    return col, row


TOGGLE_HELP_LINES = [
    "Keyboard toggles:",
    "  h  show/hide this help",
    "  d  toggle detection overlay (boxes/labels)",
    "  c  toggle corner overlay on raw frame",
    "  k  lock/unlock board transform (calibrate once)",
    "  l  toggle console log window",
    "  m  switch detection mode (model <-> diff/subtraction)",
    "  v  validate current board against the model (diff mode)",
    "  p  toggle printing board state to stdout",
    "  o  rotate square mapping 90 (view stays as raw feed)",
    "  f  toggle board flip orientation",
    "  r  start/stop recording the game (PGN + FEN log)",
    "  u  undo last recorded move + re-anchor reference",
    "  a  re-anchor reference to current view (no undo)",
    "  s  save current frame to snapshot_<n>.png",
    "  ESC  quit",
]


def square_name(col, row):
    """Map a cell of the (un-rotated) warped board grid to its square name.

    board_rotation is applied here as a coordinate rotation rather than by
    rotating the displayed image, so the on-screen view matches the raw camera
    feed while squares are still identified correctly. flip adds the 180° turn."""
    col, row = rotate_cell(col, row, settings.board_rotation)
    if settings.flip_orientation:
        col, row = 7 - col, 7 - row
    return f"{FILES[col]}{RANKS[row]}"


def detections_to_board(result, board_w, board_h, padding=0):
    """Map each detected piece to a square based on the box's bottom-center
    point (a piece's base sits on its square, the box center does not).

    Detections whose bottom-center falls in the padding margin are discarded
    so pieces near the warp edge don't get clamped onto a border square."""
    board = {}
    valid_w = board_w - 2 * padding
    valid_h = board_h - 2 * padding
    cell_w = valid_w / 8
    cell_h = valid_h / 8

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx = (x1 + x2) / 2 - padding
        cy = y2 - padding  # bottom edge of the box, offset by padding

        if not (0 <= cx < valid_w and 0 <= cy < valid_h):
            continue

        col = min(7, max(0, int(cx // cell_w)))
        row = min(7, max(0, int(cy // cell_h)))

        label = result.names[int(box.cls[0])]
        conf = float(box.conf[0])

        sq = square_name(col, row)
        # Keep the highest-confidence detection per square
        if sq not in board or conf > board[sq][1]:
            board[sq] = (label, conf)

    return board


def draw_help_overlay(frame):
    """Render the help menu as a semi-transparent box in the top-left corner."""
    out = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness, pad = 0.55, 1, 10
    line_h = 22
    box_w = 600
    box_h = pad * 2 + line_h * len(TOGGLE_HELP_LINES)
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (box_w, box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.8, out, 0.4, 0, out)
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
        out,
        "BOARD LOCKED",
        (w - 210, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


# Grid corner cells -> (text anchored left?, anchored top?) on the displayed view.
_VIEW_CORNERS = {
    (0, 0): (True, True),  # top-left
    (7, 0): (False, True),  # top-right
    (0, 7): (True, False),  # bottom-left
    (7, 7): (False, False),  # bottom-right
}


def draw_corner_markers(frame, labels=("a1", "h8")):
    """Label where the given squares sit on the board, at their live corners.

    Names come from square_name, so the markers track the current rotation/flip
    and stay correct even though the view is left in the raw-feed orientation
    (e.g. with the default mapping, a1 is bottom-right and h8 is top-left)."""
    out = frame.copy()
    h, w = out.shape[:2]
    font, scale, thickness, m = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2, 12
    wanted = {lbl.lower() for lbl in labels}
    for (col, row), (left, top) in _VIEW_CORNERS.items():
        name = square_name(col, row)
        if name not in wanted:
            continue
        text = name.upper()
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        x = m if left else w - m - tw
        y = m + th if top else h - m
        cv2.putText(out, text, (x, y), font, scale, (0, 255, 0), thickness, cv2.LINE_AA)
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
        out,
        text,
        (34, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def draw_last_move_overlay(frame, san):
    """Draw the last committed move in large text at the bottom centre."""
    out = frame.copy()
    h, w = out.shape[:2]
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3
    text = san
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (w - tw) // 2
    y = h - 18
    cv2.putText(out, text, (x + 2, y + 2), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(out, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def draw_illegal_move_overlay(frame):
    """Draw a centred 'Illegal move' warning."""
    out = frame.copy()
    h, w = out.shape[:2]
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3
    text = "Illegal move"
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (w - tw) // 2
    y = h // 2 + th // 2
    pad = 12
    overlay = out.copy()
    cv2.rectangle(overlay, (x - pad, y - th - pad), (x + tw + pad, y + pad), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)
    cv2.putText(out, text, (x, y), font, scale, (0, 0, 220), thickness, cv2.LINE_AA)
    return out


def draw_undo_prompt_overlay(frame):
    """Draw a two-line centred prompt asking the user to restore the piece first."""
    out = frame.copy()
    h, w = out.shape[:2]
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2
    lines = [
        "Move piece back to its square,",
        "then press U to confirm  (any other key cancels)",
    ]
    line_h = 36
    total_h = line_h * len(lines)
    pad = 14
    widths = [cv2.getTextSize(l, font, scale, thickness)[0][0] for l in lines]
    box_w = max(widths) + pad * 2
    box_x = (w - box_w) // 2
    box_y = h // 2 - total_h // 2 - pad
    overlay = out.copy()
    cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + total_h + pad * 2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, out, 0.35, 0, out)
    for i, line in enumerate(lines):
        tw = widths[i]
        x = (w - tw) // 2
        y = box_y + pad + (i + 1) * line_h - 6
        cv2.putText(out, line, (x, y), font, scale, (0, 200, 255), thickness, cv2.LINE_AA)
    return out


def draw_changed_overlay(warped, changed_cells, padding=0):
    """Outline the squares the subtraction step flagged as changed (diff mode)."""
    out = warped.copy()
    h, w = out.shape[:2]
    cell_w = (w - 2 * padding) / 8
    cell_h = (h - 2 * padding) / 8
    for col, row in changed_cells:
        x0 = int(padding + col * cell_w)
        x1 = int(padding + (col + 1) * cell_w)
        y0 = int(padding + row * cell_h)
        y1 = int(padding + (row + 1) * cell_h)
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 165, 255), 2)
    return out


def draw_mode_indicator(frame, mode):
    """Draw the active detection mode label, bottom-left above the REC marker."""
    out = frame.copy()
    h = out.shape[0]
    cv2.putText(
        out,
        f"mode: {mode}",
        (10, h - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 220, 220),
        2,
        cv2.LINE_AA,
    )
    return out


def draw_console_view(lines: list[str], width: int = 900) -> np.ndarray:
    """Render recent console log lines as a black image for the log overlay window."""
    line_h, pad, title_h = 20, 8, 28
    height = max(200, pad * 2 + title_h + line_h * max(1, len(lines)))
    img = np.zeros((height, width, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "Console log  (press L to hide)", (pad, pad + 14), font, 0.45, (80, 220, 80), 1, cv2.LINE_AA)
    for i, line in enumerate(lines):
        y = pad + title_h + (i + 1) * line_h
        cv2.putText(img, line[:110], (pad, y), font, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    return img


def validate_with_model(model, warped, tracker):
    """Run the YOLO model once and compare its read to the tracked position.

    The re-sync fallback for diff mode: when a change can't be explained or you
    suspect the tracker has drifted, this surfaces where the model disagrees so
    you can correct the board (the diff path never identifies pieces itself)."""
    pad = settings.warp_padding
    h_w, w_w = warped.shape[:2]
    board_crop = warped[pad:h_w - pad, pad:w_w - pad] if pad else warped
    results = model(board_crop, verbose=False, conf=settings.pieces_conf_threshold)[0]
    board = detections_to_board(
        results, board_crop.shape[1], board_crop.shape[0], padding=0
    )
    if tracker is None:
        seen = ", ".join(f"{sq}:{label}" for sq, (label, _) in sorted(board.items()))
        print(f"[validate] model sees: {seen or '(nothing)'}")
        return

    mismatches = []
    for sq_name, (label, _conf) in board.items():
        model_sym = label_to_symbol(label)
        if model_sym is None:
            continue
        try:
            piece = tracker.board.piece_at(chess.parse_square(sq_name))
        except ValueError:
            continue
        tracked_sym = piece.symbol() if piece else None
        if model_sym != tracked_sym:
            mismatches.append(
                f"{sq_name}: model={model_sym} tracked={tracked_sym or '-'}"
            )

    if mismatches:
        print("[validate] model disagrees on: " + "; ".join(sorted(mismatches)))
    else:
        print(
            "[validate] model agrees with the tracked position where it detected pieces"
        )


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
        diff_tolerance=settings.diff_tolerance,
        debug=settings.game_debug,
    )
    origin = "custom position" if start_fen else "standard opening"
    print(f"[rec] recording started from {origin} -> {pgn_path}")
    return tracker


class _ConsoleLog:
    """Tee stdout to both the terminal and a ring buffer for the console overlay."""

    def __init__(self, maxlines: int = 40):
        self._lines: deque[str] = deque(maxlen=maxlines)
        self._stdout = sys.stdout

    def write(self, text: str) -> None:
        self._stdout.write(text)
        for line in text.splitlines():
            if line.strip():
                self._lines.append(line)

    def flush(self) -> None:
        self._stdout.flush()

    def __getattr__(self, name):
        return getattr(self._stdout, name)

    @property
    def lines(self) -> list[str]:
        return list(self._lines)


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
        f"{lib}:" in build_info
        and "YES" in build_info.split(f"{lib}:")[1].split("\n")[0]
        for lib in ("GTK+", "GTK", "Qt5", "Qt6", "QT")
    )
    gui_status = "ok" if has_gui else "not detected in build info (may still work)"
    print(f"[display] DISPLAY={display!r}  OpenCV GUI: {gui_status}")


def main(start_fen=None, detection_mode=None):
    console_log = _ConsoleLog()
    sys.stdout = console_log
    _check_display()
    print("\n".join(TOGGLE_HELP_LINES))
    detection_mode = detection_mode or settings.detection_mode
    print(f"[init] detection mode: {detection_mode}")
    if detection_mode == "diff":
        print(
            "[init] diff mode: set up the start position with the whole board "
            "visible, then press 'r' to record (the board auto-locks). Press 'p' "
            "to print changed squares; the model is kept only for 'v' re-sync."
        )
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

    show_detections = settings.show_detections
    show_corners = False
    show_help = False
    show_log = False
    print_board = False
    flip = settings.flip_orientation
    snapshot_count = 0
    frame_count = 0
    tracker = None  # GameTracker while recording, else None
    board_visible = True  # tracks corner-detection state for recording logs
    locked_quad = None  # cached board quad once calibrated, else detect per frame
    reference_warped = None  # diff mode: warped board as of the last committed move
    undo_pending = False  # True after first 'u' press, waiting for confirmation
    illegal_restore_frames = 0  # consecutive clear frames since illegal_flag was set

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

        warped = None  # kept in scope for the 'v' validate handler
        if quad is not None:
            warped = warp_board(frame, quad, padding=settings.warp_padding)

            if detection_mode == "diff":
                if reference_warped is None:
                    reference_warped = warped.copy()
                scores = square_change_scores(
                    warped,
                    reference_warped,
                    padding=settings.warp_padding,
                    inner=settings.diff_inner_fraction,
                )
                changed_cells = changed_squares(scores, settings.diff_change_threshold)
                # A hand crossing the board lights up many squares at once; skip
                # those frames so they can't be matched or reset the debounce.
                noisy = len(changed_cells) > settings.diff_max_changed

                if show_detections:
                    view = draw_changed_overlay(
                        warped, changed_cells, padding=settings.warp_padding
                    )
                else:
                    view = warped

                changed_names = {square_name(c, r) for c, r in changed_cells}
                if print_board and changed_names:
                    print("changed: " + ", ".join(sorted(changed_names)))

                if tracker is not None and not noisy:
                    if tracker.update_changed(changed_names):
                        reference_warped = warped.copy()  # re-anchor to new position

                # Auto-clear the illegal flag when the piece is moved back.
                # Once changed_cells is empty for stability_frames consecutive
                # non-noisy frames the board matches the reference again, so we
                # reset the debounce and re-anchor without requiring manual 'a'.
                if tracker is not None and tracker.illegal_flag and not noisy and not changed_cells:
                    illegal_restore_frames += 1
                    if illegal_restore_frames >= settings.game_stability_frames:
                        tracker.reset_debounce()
                        reference_warped = warped.copy()
                        print("[rec] piece restored to original square — illegal move cleared")
                        illegal_restore_frames = 0
                else:
                    illegal_restore_frames = 0
            else:
                pad = settings.warp_padding
                h_w, w_w = warped.shape[:2]
                # Crop to the board interior so YOLO never runs on the padding
                # margin; detections outside the grid can't appear at all.
                board_crop = warped[pad:h_w - pad, pad:w_w - pad] if pad else warped
                results = model(
                    board_crop, verbose=False, conf=settings.pieces_conf_threshold
                )[0]

                if show_detections:
                    annotated = results.plot(font_size=4, line_width=1)
                    view = warped.copy()
                    view[pad:pad + annotated.shape[0], pad:pad + annotated.shape[1]] = annotated
                else:
                    view = warped

                board = detections_to_board(
                    results,
                    board_crop.shape[1],
                    board_crop.shape[0],
                    padding=0,
                )
                if print_board and board:
                    occupied = ", ".join(
                        f"{sq}:{label}" for sq, (label, _) in sorted(board.items())
                    )
                    print(occupied)

                if tracker is not None:
                    tracker.update(board)

        else:
            illegal_restore_frames = 0

        # The corner/quad overlay always belongs on the raw frame (matching
        # detect_corners_cv.py). corners is None when the transform is locked, so
        # only the quad outline shows then.
        if show_corners:
            view = draw_corners_overlay(frame, corners, quad)

        if locked_quad is not None:
            view = draw_lock_indicator(view)

        if tracker is not None:
            view = draw_record_overlay(view, tracker)
            if tracker.last_san:
                view = draw_last_move_overlay(view, tracker.last_san)
            if tracker.illegal_flag:
                view = draw_illegal_move_overlay(view)
        if undo_pending:
            view = draw_undo_prompt_overlay(view)

        view = draw_mode_indicator(view, detection_mode)
        view = draw_corner_markers(view)

        if show_help:
            view = draw_help_overlay(view)

        try:
            cv2.imshow(
                "Chess piece detection", scale_for_display(view, settings.display_size)
            )
        except Exception as e:
            print(f"[display] imshow failed: {type(e).__name__}: {e}", flush=True)
            break

        if show_log:
            cv2.imshow("Console log", draw_console_view(console_log.lines))

        key = cv2.waitKey(1) & 0xFF
        if frame_count == 1:
            print(f"[loop] waitKey returned {key}", flush=True)
        if undo_pending and key != 0xFF and key != ord("u"):
            undo_pending = False
            print("[rec] undo cancelled")
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
        elif key == ord("l"):
            show_log = not show_log
            print(f"[toggle] console log window: {'on' if show_log else 'off'}")
            if not show_log:
                cv2.destroyWindow("Console log")
        elif key == ord("p"):
            print_board = not print_board
            print(f"[toggle] print board state: {'on' if print_board else 'off'}")
        elif key == ord("o"):
            settings.board_rotation = (settings.board_rotation + 90) % 360
            print(f"[toggle] square mapping rotation: {settings.board_rotation}°")
        elif key == ord("f"):
            flip = not flip
            settings.flip_orientation = flip
            corner = "top-left" if flip else "bottom-right"
            print(
                f"[toggle] flip orientation: {'on' if flip else 'off'} (A1 at {corner})"
            )
        elif key == ord("m"):
            detection_mode = "diff" if detection_mode == "model" else "model"
            reference_warped = None  # re-anchor on the next warped frame
            print(f"[toggle] detection mode: {detection_mode}")
        elif key == ord("v"):
            if warped is None:
                print("[validate] no board in view to validate")
            else:
                validate_with_model(model, warped, tracker)
        elif key == ord("r"):
            if tracker is not None:
                tracker.finalize()
                print(
                    f"[rec] recording stopped ({tracker.move_count} moves) "
                    f"-> {tracker.pgn_path}"
                )
                tracker = None
            elif detection_mode == "diff" and locked_quad is None and quad is None:
                print(
                    "[rec] can't start diff recording: board not detected. Make the "
                    "whole board visible (no hands/pieces hiding the grid), then 'r'."
                )
            else:
                # Diff mode needs a stable warp: auto-lock on record start so the
                # reference image and every later frame share one transform.
                if detection_mode == "diff" and locked_quad is None:
                    locked_quad = quad.copy()
                    board_visible = True
                    print(
                        "[calib] board auto-locked for diff recording (press 'k' to unlock)."
                    )
                tracker = start_recording(start_fen)
                # Anchor the diff reference to the start position being recorded.
                reference_warped = warped.copy() if warped is not None else None
        elif key == ord("u"):
            if tracker is None:
                print("[rec] not recording")
            elif not undo_pending:
                undo_pending = True
                print("[rec] undo: move piece back to its original square, then press U again")
            else:
                undo_pending = False
                if tracker.undo_last_move() and warped is not None:
                    reference_warped = warped.copy()
        elif key == ord("a"):
            if tracker is None:
                print("[rec] not recording")
            else:
                tracker.reset_debounce()
                if warped is not None:
                    reference_warped = warped.copy()
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
    parser.add_argument(
        "--detect",
        choices=["model", "diff"],
        default=None,
        help="Detection method: 'model' (YOLO piece detector) or 'diff' "
        "(model-free image subtraction — detect which squares changed and infer "
        "the move from the rules; needs a known start and a locked board). "
        "Defaults to settings.detection_mode. Toggle at runtime with 'm'.",
    )
    return parser.parse_args()


def cli():
    """Console-script entry point (gm-detect)."""
    args = parse_args()
    start_fen = None
    if args.from_fen:
        try:
            start_fen = normalize_fen(args.from_fen)
        except ValueError as e:
            print(f"ERROR: invalid --from-fen: {e}", file=sys.stderr)
            raise SystemExit(2) from None
    main(start_fen, detection_mode=args.detect)


if __name__ == "__main__":
    cli()
