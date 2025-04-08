import os
import re

file_name = "1b1B1Qr1-7p-6r1-2P5-4Rk2-1K6-4B3-8.jpg"

def extract_file_name(image_path):
    return os.path.basename(image_path)

def parse_fen(fen_string):
    fen = fen_string.split(".")[0] 
    fen_notation = fen.replace('-', '/') + ' w - - 0 1'
    return fen_notation

