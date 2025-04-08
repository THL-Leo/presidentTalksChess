import cv2
import os
from src.chess_detector import detect_chess_pieces

# Parse a given video by seconds and return the frames
def parse_video_by_seconds(video_path, seconds):
    video = cv2.VideoCapture(video_path)
    frames = []
    fps = int(video.get(cv2.CAP_PROP_FPS))  # Get frames per second
    frame_interval = fps * seconds  # Interval to grab a frame based on seconds

    while True:
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)
        # Skip the next frames to get frames every 'seconds' interval
        for _ in range(frame_interval - 1):
            video.grab()  # Grab the frame without decoding

    video.release()
    return frames

# Get the current directory
if __name__ == "__main__":
    current_dir = os.getcwd()
    # Path to the video file
    video_path = os.path.join(current_dir, "src", "video.mov")
    
    # Parse the video and get a frame every 5 seconds
    res = parse_video_by_seconds(video_path, 5)

    output_path = os.path.join(current_dir, "src", "output_video.mp4")
    output_fps = 1  # Set FPS of output video to 1 (since you're extracting frames every 5 seconds)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for the output video
    output_video = cv2.VideoWriter(output_path, fourcc, output_fps, (res[0].shape[1], res[0].shape[0]))

    for frame in res:
        output_video.write(frame)  # Write each frame to output video

    output_video.release()
