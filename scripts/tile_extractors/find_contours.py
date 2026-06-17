import cv2
import numpy as np
from tile_extractor import ChessBoardCell, ITileExtractor

class TileExtractor(ITileExtractor):

    rectangles = None
    cells: ChessBoardCell = None

    def __init__(self):
        ITileExtractor.__init__(self)

    def extract(self, frame: cv2.Mat) -> cv2.Mat:

        if self.rectangles is None: 
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            edges = cv2.Canny(blurred, 50, 200, None, 3)

            # Fill the gaps in the edges to create a more continuous edge map
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.dilate(edges, kernel, iterations=1)

            if self.debug:
                cv2.imshow("Canny Edges", edges)

            # Detect hough lines in the warped image to draw grid
            lines = cv2.HoughLines(edges, 1, np.pi / 180, 250)

            # Create black image to draw lines on grayscale 1 channel
            line_image = np.ones_like(gray) * 255

            # Merge lines close to each other to avoid multiple lines for the same grid line
            if lines is None:
                return
        
            # Filter out lines which are not close to horizontal or vertical
            angle_threshold = np.deg2rad(1)
            filtered_lines = []
            for rho, theta in lines[:, 0]:
                if (np.abs(theta) < angle_threshold) or (np.abs(theta - np.pi / 2) < angle_threshold):
                    filtered_lines.append((rho, theta))
            lines = np.array(filtered_lines).reshape(-1, 1, 2)

            merged_lines = []
            for rho, theta in lines[:, 0]:
                if not any(abs(rho - r) < 10 and abs(theta - t) < np.deg2rad(5) for r, t in merged_lines):
                    merged_lines.append((rho, theta))
            lines = np.array(merged_lines).reshape(-1, 1, 2)

            # Draw the detected lines on the warped image
            for rho, theta in lines[:, 0]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 0), 2)

            # Draw border around the line image to avoid false positives at the edges
            border_size = 1
            line_image[:border_size, :] = 0
            line_image[-border_size:, :] = 0
            line_image[:, :border_size] = 0
            line_image[:, -border_size:] = 0

            if self.debug:
                cv2.imshow("Detected Lines", line_image)

            # Line image is a binary image with grid lines
            # Extract all the grid squares from the line image
            contours, _ = cv2.findContours(line_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
            self.rectangles = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = w / h
                if 0.8 < aspect_ratio < 1.2 and w > 20 and h > 20:  # Filter for square-like contours
                    
                    # Add small margin to the bounding box to avoid cutting off the edges of the cells
                    margin = 2  # Adjust this value as needed
                    x = max(x - margin, 0)
                    y = max(y - margin, 0)
                    w = min(w + 2 * margin, line_image.shape[1] - x)
                    h = min(h + 2 * margin, line_image.shape[0] - y)
                    self.rectangles.append((x, y, w, h))


            # Sort the rectangles from left to right, top to bottom
            self.rectangles.sort(key=lambda r: (r[1] // 50, r[0] // 50))
            self.determine_cells(frame)

        return self.cells
    
    def determine_cells(self, frame: cv2.Mat):
        # Extract all rectangles from the warped image to two dimensional list of cells
        self.cells = [[] for _ in range(8)]
        for (x, y, w, h) in self.rectangles:
            # Determine the row and column of the cell based on its center position
            col = int((x + w / 2) // w)
            row = int((y + h / 2) // h)
            chess_cell = ChessBoardCell(x, y, w, h)
            chess_cell.update(frame)
            self.cells[row].insert(col, chess_cell)


        corner_cells = [self.cells[0][0], self.cells[0][7], self.cells[7][0], self.cells[7][7]]
        
        # Check which corner is white cell if that corner is white cell then that is H1 cell
        for n, cell in enumerate(corner_cells):
            if cell.isWhite:
                break

        # Re order the cells based on the position of the H1 cell
        # Use the chess notation to determine the position of the cells in the grid
        # A is column 0, H is column 7 and 1 is row 0, 8 is row 7
        if n == 0:  # H1 is at top left corner
            # Columns need to be flipped
            for row in self.cells: row.reverse()
        elif n == 1:  # H1 is at top right corner
            self.cells.reverse()
            for row in self.cells: row.reverse()
            pass
        elif n == 2:  # H1 is at bottom left corner
            # Flip rows and columns

            pass
        elif n == 3:  # H1 is at bottom right corner
            # Flip rows
            self.cells.reverse()

        # Set the name of each cell based on its position in the grid
        for row in range(8):
            for col in range(8):
                cell = self.cells[row][col]
                cell.set_name(f"{chr(ord('A') + row)}{col+1}")

        if(self.debug):
            cv2.imshow("A1 Cell", self.cells[0][0].get_cell(frame))
            cv2.imshow("H8 Cell", self.cells[7][7].get_cell(frame))
        
        

