import cv2
import numpy as np

from chess_pieces import DEFAULT_PIECE_LOCATIONS, IChessPiece, UnknownPiece

class ChessCell:
    id: int = 0
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    row: int = 0
    col: int = 0
    name: str = ""

    piece: IChessPiece | None = None
    newPiece: IChessPiece | None = None
    
    isWhite: bool = False

    def __init__(self, x: int, y: int, width: int, height: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def __repr__(self):
        return f"ChessCell(x={self.x}, y={self.y}, width={self.width}, height={self.height}, row={self.row}, col={self.col}, name={self.name})"
    
    def init(self, row: int, col: int):
        self.row = row
        self.col = col
        self.id = row * 8 + col
        self.name = f"{chr(ord('A') + col)}{row + 1}"  # Chess notation: A1, B2, etc.
        self.piece = DEFAULT_PIECE_LOCATIONS[row][col]
    
    def update(self, frame: cv2.Mat):
        self.isWhite = self.is_white_cell(frame)

        hasPiece, isWhite = self.has_piece(frame)

        if hasPiece:
            if self.piece is not None and self.piece.isWhite == isWhite:
                self.newPiece = self.piece
            else:
                self.newPiece = UnknownPiece(isWhite)
        else:
            self.newPiece = None

        
        return self.piece != self.newPiece
    
    def canMove(self, whiteTurn: bool, cell: ChessCell):

        if self.piece == None:
            return False
        
        if self.piece.isWhite != whiteTurn:
            return False

        if cell.piece is not None and cell.piece.isWhite == self.piece.isWhite:
            return False

        y = cell.row - self.row
        x = cell.col - self.col

        return self.piece.is_valid_move(x, y, cell.piece)

    def move(self, whiteTurn: bool, cell: ChessCell):
        if self.canMove(whiteTurn, cell):
            capture = cell.piece is not None
            cell.piece = self.piece
            self.piece = None
            cell.piece.move()
            return f"{cell.piece.name[0]}{"x" if capture else ""}{cell.name}"
        return ""
        
    
    def commit(self):
        self.piece = self.newPiece

    def draw_cell(self, frame: cv2.Mat):
        color = (255, 0, 255) if self.piece is not self.newPiece else (0, 255, 0) if self.piece is not None else (255, 0, 0)
        margin = 5
        self.has_piece(frame,True)
        cv2.rectangle(frame, (self.x + margin, self.y + margin), (self.x + self.width - margin, self.y + self.height - margin), color, 2)
        cv2.putText(frame, self.name, (self.x + 5, self.y + 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)
       
        if self.piece is not None:
            cv2.putText(frame, self.piece.name, (self.x + 5, self.y + 40), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255),1)

   

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

    
    def has_piece(self, frame: cv2.Mat, draw: bool = False):
        cell = self.get_cell(frame)
        
        gray = cv2.cvtColor(cell, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        canny = cv2.Canny(blurred, 50, 100, apertureSize=3)

        # Check if there are any contours in the cell, if there are then there is a piece on the cell
        # Only check in the center of the cell to avoid detecting the edges of the cell as pieces
        padding = int(self.width / 4)
        contours, _ = cv2.findContours(canny[padding:-padding, padding:-padding], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        isWhite = False

        if len(contours) > 0:
            all_points = np.vstack(contours)
            merged = cv2.convexHull(all_points)

            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

            piece = np.zeros_like(gray)
            cv2.drawContours(piece[padding:-padding, padding:-padding], [merged], -1, (255), cv2.FILLED)
            piece = cv2.bitwise_and(thresh, piece)


      
            white_ratio = cv2.countNonZero(piece) / (cell.shape[0] * cell.shape[1])
            isWhite = white_ratio > 0.05

            if draw:
                cv2.drawContours(cell[padding:-padding, padding:-padding], [merged], -1, (255, 255, 255) if isWhite else (0,0,0), cv2.FILLED)

            return (True, isWhite)

        return (False, isWhite) 
    
    