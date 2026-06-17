import cv2

from chessboard_extractors.find_chessboard import ChessboardExtractor
from chessboard_extractor import IChessboardExtractor

from tile_extractor import ITileExtractor
from tile_extractors.find_contours import TileExtractor


def main():
    # Create an instance of the chessboard extractor
    extractor: IChessboardExtractor = ChessboardExtractor()
    
    tile_extractor: ITileExtractor = TileExtractor()
    tile_extractor.debug = True  # Enable debug mode to visualize the tile extraction process

    # Replace with your stream URL
    stream_url = "http://10.42.0.177:4444/stream"
    cap = cv2.VideoCapture(stream_url)
    cell_updated_frame_count = 0
    instruction_window = cv2.namedWindow("Instruction")


    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from stream.")
            break

        # Extract the chessboard from the image
        success, extracted_chessboard = extractor.extract(frame)

        if not success:
            print("Chessboard not found in the frame.")
            continue

        # Extract the tiles from the chessboard
        cells = tile_extractor.extract(extracted_chessboard)


        if cells is not None:
            update_cells = []

            for row in cells:
                for cell in row:
                    if cell.update(extracted_chessboard):
                        update_cells.append(cell)
                    cell.draw_cell(extracted_chessboard)
            
            if len(update_cells) == 2 and update_cells[0].new_state != update_cells[1].new_state:
                cell_updated_frame_count += 1
                if cell_updated_frame_count > 10:
                    print(f"Cells updated: {update_cells[0].name} and {update_cells[1].name}")
                    update_cells[0].commit_state()
                    update_cells[1].commit_state()
                    cell_updated_frame_count = 0

                    # Write cell updated text to new opencv window



            elif len(update_cells) > 2:
                cell_updated_frame_count = 0


        # Display the extracted chessboard
        cv2.imshow("Extracted Chessboard", extracted_chessboard)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture
    cap.release()
    cv2.destroyAllWindows()


# Main method
if __name__ == "__main__":
    main()