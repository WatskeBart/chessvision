import cv2
import numpy as np

from chessvision.settings import settings

# A standard 8×8 board has 7×7 internal corner intersections.
INNER_CORNERS = (7, 7)

SUBPIX_CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001,
)


def find_corners(gray):
    """Return the 7×7 internal grid corners, or None if detection fails.

    cv2.findChessboardCornersSB is OpenCV's newer, more robust variant that
    handles poor lighting and partial occlusion better than the classic
    findChessboardCorners, and does its own sub-pixel refinement internally.
    Falls back to the classic detector + cornerSubPix if SB fails."""
    ok, corners = cv2.findChessboardCornersSB(
        gray,
        INNER_CORNERS,
        flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE,
    )
    if ok:
        return corners

    ok, corners = cv2.findChessboardCorners(
        gray,
        INNER_CORNERS,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if ok:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), SUBPIX_CRITERIA)
    return corners if ok else None


def board_quad_from_corners(corners):
    """Derive the outer board quad from the 7×7 internal corner grid.

    The detected corners cover only the internal 7×7 intersections. We
    extrapolate one square outward on all four sides to reach the board edge."""
    pts = corners.reshape(INNER_CORNERS[1], INNER_CORNERS[0], 2)

    # One-square step vectors estimated from the grid edges.
    step_right = (pts[0, -1] - pts[0, 0]) / (INNER_CORNERS[0] - 1)
    step_down = (pts[-1, 0] - pts[0, 0]) / (INNER_CORNERS[1] - 1)

    tl = pts[0, 0] - step_right - step_down
    tr = pts[0, -1] + step_right - step_down
    br = pts[-1, -1] + step_right + step_down
    bl = pts[-1, 0] - step_right + step_down

    return np.array([tl, tr, br, bl], dtype="float32")


def warp_board(frame, quad, size=800, padding=0):
    """Perspective-warp the detected quad to a square top-down view.

    padding adds extra pixels of original context around each edge so pieces
    that overhang the board boundary are still visible in the output image."""
    p = padding
    dst = np.array(
        [[p, p], [size + p - 1, p], [size + p - 1, size + p - 1], [p, size + p - 1]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(frame, M, (size + 2 * p, size + 2 * p))


def main():
    """Stream preview using OpenCV's native chessboard corner detector."""
    url = str(settings.stream_url)
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open stream: {url}")

    warp_enabled = True

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream ended or frame read failed.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = find_corners(gray)

        view = frame.copy()
        if corners is not None:
            cv2.drawChessboardCorners(view, INNER_CORNERS, corners, True)
            if warp_enabled:
                quad = board_quad_from_corners(corners)
                try:
                    view = warp_board(
                        frame, quad,
                        size=settings.display_size,
                        padding=settings.warp_padding,
                    )
                except cv2.error:
                    pass  # degenerate quad; keep the annotated raw frame

        status = "warp: ON" if warp_enabled else "warp: OFF"
        detected = "corners: OK" if corners is not None else "corners: --"
        cv2.putText(view, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(view, detected, (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        cv2.imshow("CV chessboard corners", view)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        if key == ord("w"):
            warp_enabled = not warp_enabled

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
