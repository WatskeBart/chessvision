import cv2
import numpy as np

from settings import settings


def _merge_collinear(lines, rho_thresh, theta_thresh=np.radians(5)):
    """Collapse near-duplicate (rho, theta) Hough lines into one averaged
    line each. A single grid line (the edge shared by two squares) is
    typically detected several times, slightly offset by noise."""
    merged = []
    for rho, theta in sorted(lines, key=lambda line: line[0]):
        for i, (mrho, mtheta, count) in enumerate(merged):
            if abs(rho - mrho) < rho_thresh and abs(theta - mtheta) < theta_thresh:
                merged[i] = (
                    (mrho * count + rho) / (count + 1),
                    (mtheta * count + theta) / (count + 1),
                    count + 1,
                )
                break
        else:
            merged.append((rho, theta, 1))
    return [(rho, theta) for rho, theta, _ in merged]


def _line_intersection(line1, line2):
    """Intersect two lines given in Hough (rho, theta) normal form."""
    rho1, theta1 = line1
    rho2, theta2 = line2
    a = np.array(
        [
            [np.cos(theta1), np.sin(theta1)],
            [np.cos(theta2), np.sin(theta2)],
        ]
    )
    b = np.array([rho1, rho2])
    if abs(np.linalg.det(a)) < 1e-6:
        return None
    return np.linalg.solve(a, b)


def _find_board_corners_hough(edges):
    """Find the board the way a checkerboard calibration target is found:
    detect the straight lines formed by the rank/file square edges with a
    Hough transform, split them into two perpendicular bundles, and take the
    outermost line on each side as the board's boundary. Robust to a broken
    or partially occluded outer border, as long as enough internal square
    edges are visible to fix the grid's extent."""
    h, w = edges.shape
    min_vote = int(min(h, w) * 0.3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=min_vote)
    if lines is None or len(lines) < 4:
        return None

    verticals, horizontals = [], []
    for rho, theta in lines[:, 0]:
        theta_deg = np.degrees(theta)
        # theta near 0/180 -> a vertical line (a file edge); theta near 90 ->
        # a horizontal line (a rank edge).
        if theta_deg < 45 or theta_deg > 135:
            verticals.append((rho, theta))
        else:
            horizontals.append((rho, theta))

    rho_thresh = min(h, w) * 0.02
    verticals = _merge_collinear(verticals, rho_thresh)
    horizontals = _merge_collinear(horizontals, rho_thresh)
    if len(verticals) < 2 or len(horizontals) < 2:
        return None

    verticals.sort(key=lambda line: line[0])
    horizontals.sort(key=lambda line: line[0])
    left, right = verticals[0], verticals[-1]
    top, bottom = horizontals[0], horizontals[-1]

    corners = [
        _line_intersection(top, left),
        _line_intersection(top, right),
        _line_intersection(bottom, right),
        _line_intersection(bottom, left),
    ]
    if any(c is None for c in corners):
        return None
    return np.array(corners, dtype="float32")


def find_board_corners(frame):
    """Locate the board's outer quadrilateral in the raw camera frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    threshold_kernel_size = blurred.shape[0] // 32 * 2 + 1
    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        threshold_kernel_size,
        3,
    )
    edges = cv2.Canny(thresh, 50, 200, None, 3)

    corners = _find_board_corners_hough(edges)
    if corners is not None:
        return corners

    # Fall back to the single outer-contour approach if the grid lines
    # couldn't be resolved (e.g. low contrast, or too few squares visible).
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(frame, M, (max_width, max_height))
    return warped


def main():
    """Standalone preview: warp the detected board out of the live stream."""
    cap = cv2.VideoCapture(str(settings.stream_url))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        corners = find_board_corners(frame)
        view = frame
        if corners is not None:
            cv2.drawContours(frame, [corners.astype(np.int32)], -1, (0, 255, 0), 2)
            view = warp_board(frame, corners)

        cv2.imshow("Chessboard detection", view)
        if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
