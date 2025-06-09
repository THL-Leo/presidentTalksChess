import os
import cv2
import asyncio
from src.video_parser import parse_video_by_seconds
from src.chess_detector import process_frames

async def main():
    # Get the current directory
    current_dir = os.getcwd()
    
    # Path to the video file
    video_path = os.path.join(current_dir, "src", "video.mov")
    
    # Path to the output directory
    output_dir = os.path.join(current_dir, "src", "output_video.mp4")
    
    # Parse the video by seconds
    frames = parse_video_by_seconds(video_path, 2)
    
    # Process all frames with a single engine instance
    fens = await process_frames(frames)
    
    return fens

if __name__ == "__main__":
    try:
        fens = asyncio.run(main())
        print("Final FEN sequence:")
        for i, fen in enumerate(fens):
            print(f"Frame {i+1}: {fen}")
    except Exception as e:
        print(f"Error in main: {e}")