

class IChessPiece(object):
    isWhite: bool
    name: str

    def __init__(self, name: str, isWhite: bool):
        self.name = name
        self.isWhite = isWhite

    def is_valid_move(self, x: int, y: int, attack: IChessPiece) -> bool:
        raise Exception("NotImplementedException")
    
    def move(self):
        pass
    
    def __repr__(self):
        return f"{self.name}"
    
class UnknownPiece(IChessPiece):
    def __init__(self, isWhite: bool):
        super().__init__("Unknown", isWhite)

    def is_valid_move(self, x: int, y: int, attack: IChessPiece) -> bool:
        return False  # Unknown pieces can move anywhere, but this should never be called in practice.
 
class Pawn(IChessPiece):
    is_first_move = True

    def __init__(self, isWhite: bool):
        super().__init__("Pawn", isWhite)

    def is_valid_move(self, x: int, y: int, attack: IChessPiece) -> bool:
        # Pawns can only move forward one square, except on their first move when they can move forward two squares.
        # Pawns can only attack diagonally one square forward.
        if not self.isWhite:
            y *=-1

        if attack is not None:
            return abs(x) == 1 and y == 1
        else:
            return x == 0 and (y == 1 or (y == 2 and self.is_first_move))
        
    def move(self):
        self.is_first_move = False
        pass

class Rook(IChessPiece):
    def __init__(self, isWhite: bool):
        super().__init__("Rook", isWhite)

    def is_valid_move(self, x: int, y: int, attack: IChessPiece) -> bool:
        # Rooks can move any number of squares along a rank or file, but cannot leap over other pieces.
        return (x == 0 or y == 0) and (x != 0 or y != 0)

class Knight(IChessPiece):
    def __init__(self, isWhite: bool):
        super().__init__("Knight", isWhite)

    def is_valid_move(self, x: int, y: int, attack: IChessPiece) -> bool:
        # Knights move in an "L" shape: two squares in a horizontal or vertical direction, then one square perpendicular to that.
        return (abs(x) == 2 and abs(y) == 1) or (abs(x) == 1 and abs(y) == 2)
    

class Bishop(IChessPiece):
    def __init__(self, isWhite: bool):
        super().__init__("Bishop", isWhite)

    def is_valid_move(self, x: int, y: int, attack: IChessPiece) -> bool:
        # Bishops can move any number of squares diagonally, but cannot leap over other pieces.
        return abs(x) == abs(y) and (x != 0 or y != 0)
    

class Queen(IChessPiece):
    def __init__(self, isWhite: bool):
        super().__init__("Queen", isWhite)

    def is_valid_move(self, x: int, y: int, attack: IChessPiece) -> bool:
        # Queens can move any number of squares along a rank, file, or diagonal, but cannot leap over other pieces.
        return (x == 0 or y == 0 or abs(x) == abs(y)) and (x != 0 or y != 0)
    
class King(IChessPiece):
    def __init__(self, isWhite: bool):
        super().__init__("King", isWhite)

    def is_valid_move(self, x: int, y: int, attack: IChessPiece) -> bool:
        # Kings can move one square in any direction.
        return abs(x) <= 1 and abs(y) <= 1 and (x != 0 or y != 0)
    
DEFAULT_PIECE_LOCATIONS: list[list[IChessPiece | None]] = [
    [Rook(True), Knight(True), Bishop(True), Queen(True), King(True), Bishop(True), Knight(True), Rook(True)],
    [Pawn(True), Pawn(True), Pawn(True), Pawn(True), Pawn(True), Pawn(True), Pawn(True), Pawn(True)],
    [None, None, None, None, None, None, None, None],
    [None, None, None, None, None, None, None, None],
    [None, None, None, None, None, None, None, None],
    [None, None, None, None, None, None, None, None],
    [Pawn(False), Pawn(False), Pawn(False), Pawn(False), Pawn(False), Pawn(False), Pawn(False), Pawn(False)],
    [Rook(False), Knight(False), Bishop(False), Queen(False), King(False), Bishop(False), Knight(False), Rook(False)]
]