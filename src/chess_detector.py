import torch
import os
import cv2
import chess
import chess.engine
import time
from ultralytics import YOLO

PIECE_CLASS_MAPPING = {
    'K': 0, 'Q': 1, 'R': 2, 'B': 3, 'N': 4, 'P': 5,
    'k': 6, 'q': 7, 'r': 8, 'b': 9, 'n': 10, 'p': 11
}

def stockfish_check_orientation(fen, engine):
    """Analyze a position using Stockfish and return its evaluation score."""
    try:
        board = chess.Board(fen)
        # Use a very limited depth to avoid engine crashes
        info = engine.analyse(board, chess.engine.Limit(depth=5, time=0.5))
        score = info["score"].relative
        
        if score.is_mate():
            return 100000 if score.mate() > 0 else -100000
        else:
            return score.score(mate_score=10000)
    except Exception as e:
        print(f"Error in analysis: {e}")
        return 0  # Default score if analysis fails

def correct_fen_if_flipped(fen):
    """Flip the chess FEN notation if board orientation is incorrect."""
    parts = fen.split(" ")
    board_part = parts[0]
    ranks = board_part.split("/")
    
    # Simply reverse the ranks (vertical flip)
    flipped_ranks = ranks[::-1]
    flipped_board = "/".join(flipped_ranks)
    
    # Keep the rest of the FEN the same
    flipped_fen = flipped_board + " " + " ".join(parts[1:])
    return flipped_fen

def detect_chess_pieces(img=None, img_path=None, prev_state=None, engine=None, count = 0):
    """Detect chess pieces on the board and determine correct FEN orientation."""
    current_dir = os.getcwd()
    model_path = os.path.join(current_dir, "model", "best.pt")
    
    if img_path:
        img = cv2.imread(img_path)
    
    if img_path is None and img is None:
        raise ValueError("No valid image provided. Either pass 'img' or a valid 'img_path'.")
    
    # img = cv2.resize(img, (400, 400))  # Resize to 400x400
    model = YOLO(model_path)
    results = model(img, conf=0.8)


    board = chess.Board()
    board.clear_board()

    # Add detected pieces to the board
    for result in results:
        for i, box in enumerate(result.boxes):
            if i > 31:  # Stop after detecting 32 pieces
                break
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())

            x = (x1 + x2) / 2
            y = (y1 + y2) / 2
            x_scale = (x2 - x1) / 50
            y_scale = (y2 - y1) / 50

            piece = list(PIECE_CLASS_MAPPING.keys())[cls]
            square = chess.square(int((x / x_scale) // 50), 7 - int((y / y_scale) // 50))
            if not board.piece_at(square):
                board.set_piece_at(square, chess.Piece.from_symbol(piece))
        # result.save(filename=f"result{count}.jpg")

    # Generate new FEN
    fen_parts = board.fen().split()

    # Maintain previous move counters
    if prev_state:
        prev_parts = prev_state.split()
        fen_parts[1] = 'b' if prev_parts[1] == 'w' else 'w'  # Toggle turn
        if fen_parts[1] == 'w':  # Black just moved, increment full-move counter
            fen_parts[5] = str(int(prev_parts[5]) + 1)
        else:
            fen_parts[5] = prev_parts[5]  # Keep the same move count for White

    new_fen = " ".join(fen_parts)
    print(f"Original FEN: {new_fen}")
    
    # Skip Stockfish orientation check if engine isn't provided
    if engine is None:
        return new_fen
    try:
        # Simple orientation check - we'll just try the original and flipped orientation
        score = stockfish_check_orientation(new_fen, engine)
        flipped_fen = correct_fen_if_flipped(new_fen)
        print(f"Flipped FEN: {flipped_fen}")
        flipped_score = stockfish_check_orientation(flipped_fen, engine)
        
        # Choose the FEN with the better score
        final_fen = flipped_fen if abs(flipped_score) < abs(score) else new_fen
        print(f'Final Output: {final_fen}')
        return final_fen
    except Exception as e:
        print(f"Error during orientation check: {e}")
        # If something goes wrong, return the original FEN
        return new_fen

def process_frames(frames, prev_fen=None):
    """Process multiple frames with a single engine instance."""
    engine_path = "./stockfish/stockfish-macos-m1-apple-silicon"
    
    # Start the engine just once
    try:
        engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        print("Engine started successfully")
        
        fens = []
        for i, frame in enumerate(frames):
            print(f"Processing frame {i+1}/{len(frames)}")
            try:
                fen = detect_chess_pieces(
                    img=frame, 
                    prev_state=prev_fen if prev_fen else None,
                    engine=engine,
                    count = i
                )
                prev_fen = fen
                fens.append(fen)
            except Exception as e:
                print(f"Error processing frame {i}: {e}")
                # If one frame fails, continue with the next
                if prev_fen:
                    fens.append(prev_fen)
                else:
                    fens.append("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")  # Default position
                    
        return fens
    except Exception as e:
        print(f"Error starting engine: {e}")
        # Process without engine if it fails to start
        fens = []
        for i, frame in enumerate(frames):
            fen = detect_chess_pieces(
                img=frame, 
                prev_state=prev_fen if prev_fen else None,
                engine=None,  # No engine
                count = i
            )
            prev_fen = fen
            fens.append(fen)
        return fens
    finally:
        # Make sure to quit the engine
        if 'engine' in locals():
            try:
                engine.quit()
                print("Engine terminated successfully")
            except Exception as e:
                print(f"Error terminating engine: {e}")

if __name__ == "__main__":
    process_frames(frames=[cv2.imread('src/board.png')], prev_fen=None)
    # expect 3r1rk1/5qp1/b2bp3/2pp2Pp/8/1P2PN1P/PB3KB1/R2QR3 b - - 0 20
