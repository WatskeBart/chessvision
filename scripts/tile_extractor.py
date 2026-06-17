import cv2
import numpy as np

class ITileExtractor(object):

    debug = False

    def __init__(self):
        pass

    def extract(self, frame: cv2.Mat) -> cv2.Mat:
        raise Exception("NotImplementedException")
    

class ChessBoardCellState:
    EMPTY = 0
    OCCUPIED = 1
    UNKNOWN = 2
    

class ChessBoardCell:

    x = 0
    y = 0
    width = 0
    height = 0
    state = ChessBoardCellState.UNKNOWN
    new_state = ChessBoardCellState.UNKNOWN
    isWhite = False
    name = "UNKNOWN"



    def __init__(self, x: int, y: int, width: int, height: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def __repr__(self):
        return f"ChessBoardCell(x={self.x}, y={self.y}, width={self.width}, height={self.height})"
    
    def update(self, frame: cv2.Mat):
        cell = self.get_cell(frame)
        self.isWhite = self.is_white_cell(frame)


        self.new_state = ChessBoardCellState.OCCUPIED if self.has_piece(frame) else ChessBoardCellState.EMPTY
        if self.state == ChessBoardCellState.UNKNOWN:
            self.state = self.new_state
        
        return self.state != self.new_state
    
    def commit_state(self):
        self.state = self.new_state

    def draw_cell(self, frame: cv2.Mat):
        color = (255, 0, 0) if self.state == ChessBoardCellState.OCCUPIED else (0, 255, 0)
        margin = 5
        cv2.rectangle(frame, (self.x + margin, self.y + margin), (self.x + self.width - margin, self.y + self.height - margin), color, 2)


    def get_cell(self, frame: cv2.Mat):
        return frame[self.y:self.y+self.height, self.x:self.x+self.width]

    def is_white_cell(self, frame: cv2.Mat):
        cell = self.get_cell(frame)
        gray = cv2.cvtColor(cell, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        white_ratio = cv2.countNonZero(thresh) / (cell.shape[0] * cell.shape[1])
        return white_ratio > 0.75
    
    def is_black_cell(self, frame: cv2.Mat):
        cell = self.get_cell(frame)
        gray = cv2.cvtColor(cell, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 0, 150, cv2.THRESH_BINARY)
        black_ratio = cv2.countNonZero(thresh) / (cell.shape[0] * cell.shape[1])
        return black_ratio > 0.75
    
    def set_name(self, name: str):
        self.name = name
    
    def has_piece(self, frame: cv2.Mat):
        cell = self.get_cell(frame)
        
        gray = cv2.cvtColor(cell, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        canny = cv2.Canny(blurred, 50, 150, apertureSize=3)

        # Check if there are any contours in the cell, if there are then there is a piece on the cell
        # Only check in the center of the cell to avoid detecting the edges of the cell as pieces
        padding = 10
        contours, _ = cv2.findContours(canny[padding:-padding, padding:-padding], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        cv2.drawContours(cell[padding:-padding, padding:-padding], contours, -1, (0, 255, 0), 2)


        if len(contours) > 0:
            return True

        return False
    
    