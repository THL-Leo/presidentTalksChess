import chess
import os

from src.utils import extract_file_name, parse_fen

PIECE_CLASS_MAPPING = {
    'K': 0, 'Q': 1, 'R': 2, 'B': 3, 'N': 4, 'P': 5,
    'k': 6, 'q': 7, 'r': 8, 'b': 9, 'n': 10, 'p': 11
}

def generate_yolo_labels(filename, output_path):
    name = extract_file_name(filename)
    fen = parse_fen(name)
    board = chess.Board(fen)
    labels = []

    square_size = 50
    board_size = 400

    for rank in range(8):
        for file in range(8):
            piece = board.piece_at(chess.square(file, 7 - rank))
            if piece:
                class_type = PIECE_CLASS_MAPPING[str(piece.symbol())]
                x_pos = (file * square_size + square_size / 2) / board_size
                y_pos = (rank * square_size + square_size / 2) / board_size
                width = square_size / board_size
                height = square_size / board_size
                labels.append(f"{class_type} {x_pos:.6f} {y_pos:.6f} {width:.6f} {height:.6f}")

    with open(output_path, 'w') as f:
        f.write("\n".join(labels))

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # Get project root

    train_path = os.path.join(base_dir, 'data', 'train', 'images')
    test_path = os.path.join(base_dir, 'data', 'test', 'images')
    val_path = os.path.join(base_dir, 'data', 'val', 'images')
    output_train_path = os.path.join(base_dir, 'data', 'train', 'labels')
    output_test_path = os.path.join(base_dir, 'data', 'test', 'labels')
    output_val_path = os.path.join(base_dir, 'data', 'val', 'labels')

    for filename in os.listdir(train_path):
        if filename.endswith('.jpeg'):
            input_path = os.path.join(train_path, filename)
            output_path = os.path.join(output_train_path, filename.replace('.jpeg', '.txt'))
            generate_yolo_labels(input_path, output_path)
    for filename in os.listdir(test_path):
        if filename.endswith('.jpeg'):
            input_path = os.path.join(test_path, filename)
            output_path = os.path.join(output_test_path, filename.replace('.jpeg', '.txt'))
            generate_yolo_labels(input_path, output_path)

    # for filename in os.listdir(val_path):
    #     if filename.endswith('.jpeg'):
    #         input_path = os.path.join(val_path, filename)
    #         output_path = os.path.join(output_val_path, filename.replace('.jpeg', '.txt'))
    #         generate_yolo_labels(input_path, output_path)
    print("YOLO labels generated successfully.")