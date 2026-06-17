import cv2
import numpy as np
from chessboard_extractor import IChessboardExtractor

class ChessboardExtractor(IChessboardExtractor):

    board_size = (7, 7)
    output_size = (640, 640)
    padding = 10
    board = None

    def __init__(self):
        IChessboardExtractor.__init__(self)

    def extract(self, frame: cv2.Mat):


        if self.board is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            ok, corners = cv2.findChessboardCornersSB(blurred, self.board_size, flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE)

            if ok:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                pts = corners2.reshape(self.board_size[1], self.board_size[0], 2)

                # One-square step vectors estimated from the grid edges.
                step_right = (pts[0, -1] - pts[0, 0]) / (self.board_size[0] - 1)
                step_down = (pts[-1, 0] - pts[0, 0]) / (self.board_size[1] - 1)

                tl = pts[0, 0] - step_right - step_down
                tr = pts[0, -1] + step_right - step_down
                br = pts[-1, -1] + step_right + step_down
                bl = pts[-1, 0] - step_right + step_down

                self.board = np.array([
                    tl, tr, br, bl
                ], dtype="float32")

        if self.board is not None:
            dst = np.array([
                [self.padding, self.padding], 
                [self.output_size[0] + self.padding - 1, self.padding], 
                [self.output_size[0] + self.padding - 1, self.output_size[1] + self.padding - 1], 
                [self.padding, self.output_size[1] + self.padding - 1]
            ], dtype="float32")

            M = cv2.getPerspectiveTransform(self.board, dst)
            return True, cv2.warpPerspective(frame, M, (self.output_size[0] + 2 * self.padding, self.output_size[1] + 2 * self.padding))

        return False, frame
        