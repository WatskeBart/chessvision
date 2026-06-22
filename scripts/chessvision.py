import cv2
import numpy as np

from chessboard_extractors.find_chessboard import ChessboardExtractor
from chessboard_extractor import IChessboardExtractor

from chess_cell import ChessCell
from tile_extractor import ITileExtractor
from tile_extractors.find_contours import TileExtractor


def main():
    # Create an instance of the chessboard extractor
    extractor: IChessboardExtractor = ChessboardExtractor()
    
    tile_extractor: ITileExtractor = TileExtractor()
   # tile_extractor.debug = True  # Enable debug mode to visualize the tile extraction process

    # Replace with your stream URL
    stream_url = "http://10.42.0.177:4444/stream"
    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return
    

    updatedCells: list[ChessCell] = []
    updatedCellsCount: int = 0
    whiteTurn = True

    screen = None

    last_move = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from stream.")
            break

        # Extract the chessboard from the image
        success, extracted_chessboard = extractor.extract(frame)

        if not success:
            print("Chessboard not found in the frame.")
            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Extract the tiles from the chessboard
        cells = tile_extractor.extract(extracted_chessboard)


        if cells is not None:

            prevUpdatedCells = updatedCells
            updatedCells = []

            for row in cells:
                for cell in row:
                    if cell.update(extracted_chessboard):
                        updatedCells.append(cell)
                    cell.draw_cell(extracted_chessboard)

            if len(prevUpdatedCells) == len(updatedCells):
                updatedCellsCount += 1
            else:
                updatedCellsCount = 0


            moved: list[int] = []



            if updatedCellsCount > 10 and len(updatedCells) > 1 and len(updatedCells) < 5:
                for cellA in updatedCells:
                    
                    if cellA.id in moved: 
                        continue

                    for cellB in updatedCells:

                        if cellA.id == cellB.id or cellB in moved:
                            continue
                     
                        if cellA.canMove(whiteTurn, cellB):
                            last_move = cellA.move(whiteTurn, cellB)
                        elif cellB.canMove(whiteTurn, cellA):
                            last_move = cellB.move(whiteTurn, cellA)
                        else:
                            continue

                        moved.append(cellA.id)
                        moved.append(cellB.id)
                        whiteTurn = not whiteTurn
            elif updatedCellsCount > 0 and len(updatedCells) > 24:
                extractor.release()
                tile_extractor.release()
        else:
            extractor.release()


        
        if screen is None:
            screen = np.ndarray((extracted_chessboard.shape[0] + 100,) + extracted_chessboard.shape[1:], dtype= extracted_chessboard.dtype)
        
        screen[:extracted_chessboard.shape[0], :] = extracted_chessboard
        screen[extracted_chessboard.shape[0]:, :] = (255, 255, 255)


        botton_screen_shape = screen[extracted_chessboard.shape[0]:, :].shape
        
        if last_move is not None:
            cv2.putText(screen[extracted_chessboard.shape[0]:, :], last_move, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 1)



        # Display the extracted chessboard
        cv2.imshow("Game", screen)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture
    cap.release()
    cv2.destroyAllWindows()


# Main method
if __name__ == "__main__":
    main()