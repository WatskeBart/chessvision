import cv2
import numpy as np
from chessboard_extractor import IChessboardExtractor

class ChessboardExtractor(IChessboardExtractor):

    board_size = (7, 7)
    output_size = (640, 640)
    padding = 10
    board = None
    preview = None

    def __init__(self):
        IChessboardExtractor.__init__(self)

    def extract(self, frame: cv2.Mat):

        self.preview = frame.copy()

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

                cv2.drawChessboardCorners(self.preview, self.board_size, corners, ok)

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

            # Draw rectangle on preview
            cv2.polylines(self.preview, np.array([self.board], dtype="int32"), True, (255, 0, 0))

            cv2.imshow("Preview", self.preview)

            M = cv2.getPerspectiveTransform(self.board, dst)
            return True, cv2.warpPerspective(frame, M, (self.output_size[0] + 2 * self.padding, self.output_size[1] + 2 * self.padding))
        
        cv2.imshow("Preview", self.preview)
        

        return False, frame
    
    def release(self):
        self.board = None
        