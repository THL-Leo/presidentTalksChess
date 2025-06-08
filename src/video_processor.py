import os
import cv2
import subprocess
import json
from typing import List, Dict, Tuple
from src.video_parser import parse_video_by_seconds
from src.chess_detector import process_frames
from src.chess_commentator import ChessCommentator, CommentaryEvent
from src.presidential_tts import generate_presidential_commentary_sync
import tempfile

class ChessVideoProcessor:
    def __init__(self, stockfish_path: str = "./stockfish/stockfish-macos-m1-apple-silicon"):
        self.stockfish_path = stockfish_path
        self.temp_dir = None
        
    def setup_temp_directory(self):
        """Create temporary directory for processing"""
        self.temp_dir = tempfile.mkdtemp(prefix="chess_video_")
        print(f"Working directory: {self.temp_dir}")
        return self.temp_dir

    def cleanup_temp_directory(self):
        """Clean up temporary files"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
            print("Cleaned up temporary files")

    def extract_video_info(self, video_path: str) -> Dict:
        """Extract video metadata using ffprobe"""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json', 
                '-show_format', '-show_streams', video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                info = json.loads(result.stdout)
                
                # Extract video stream info
                video_stream = None
                for stream in info['streams']:
                    if stream['codec_type'] == 'video':
                        video_stream = stream
                        break
                
                if video_stream:
                    return {
                        'duration': float(info['format']['duration']),
                        'fps': eval(video_stream['r_frame_rate']),  # e.g., "30/1" -> 30.0
                        'width': video_stream['width'],
                        'height': video_stream['height'],
                        'has_audio': any(s['codec_type'] == 'audio' for s in info['streams'])
                    }
            return {}
        except Exception as e:
            print(f"Error extracting video info: {e}")
            return {}

    def parse_video_frames(self, video_path: str, frame_interval: int = 2) -> Tuple[List, List[float]]:
        """Parse video and extract frames with timestamps"""
        print(f"Parsing video: {video_path}")
        print(f"Extracting frames every {frame_interval} seconds...")
        
        # Get video info
        video_info = self.extract_video_info(video_path)
        fps = video_info.get('fps', 30)
        
        # Extract frames
        frames = parse_video_by_seconds(video_path, frame_interval)
        
        # Calculate timestamps
        timestamps = [i * frame_interval for i in range(len(frames))]
        
        print(f"Extracted {len(frames)} frames")
        return frames, timestamps

    def analyze_chess_positions(self, frames: List, timestamps: List[float]) -> List[str]:
        """Analyze chess positions from video frames"""
        print("Analyzing chess positions...")
        
        # Use existing chess detector
        fens = process_frames(frames)
        
        print(f"Analyzed {len(fens)} positions")
        return fens

    def generate_commentary(self, fens: List[str], timestamps: List[float], 
                          openai_api_key: str = None) -> List[CommentaryEvent]:
        """Generate chess commentary for the positions"""
        print("Generating chess commentary...")
        
        commentator = ChessCommentator(self.stockfish_path, openai_api_key)
        events = commentator.analyze_game_sequence(fens, timestamps)
        
        print(f"Generated {len(events)} commentary events")
        return events

    def create_audio_files(self, events: List[CommentaryEvent], president: str, 
                          output_dir: str) -> List[Dict]:
        """Generate presidential voice audio files"""
        print(f"Generating {president} voice commentary...")
        
        audio_files = generate_presidential_commentary_sync(
            events, president, output_dir
        )
        
        print(f"Generated {len(audio_files)} audio files")
        return audio_files

    def mute_original_video(self, input_video: str, output_video: str) -> bool:
        """Remove audio from original video"""
        try:
            cmd = [
                'ffmpeg', '-i', input_video, '-c:v', 'copy', '-an', 
                '-y', output_video
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
        except Exception as e:
            print(f"Error muting video: {e}")
            return False

    def create_audio_timeline(self, audio_files: List[Dict], video_duration: float) -> str:
        """Create a combined audio track with proper timing"""
        if not audio_files:
            return None
        
        # Create filter complex for ffmpeg
        filter_parts = []
        input_files = []
        
        # Add silence as base track
        filter_parts.append(f"aevalsrc=0:duration={video_duration}:sample_rate=22050[silence]")
        
        current_input = "[silence]"
        
        for i, audio_file in enumerate(audio_files):
            input_files.extend(['-i', audio_file['audio_path']])
            
            # Mix this audio at the specified timestamp
            filter_parts.append(
                f"{current_input}[{i+1}:0]adelay={int(audio_file['timestamp']*1000)}|"
                f"{int(audio_file['timestamp']*1000)}[delayed{i}]"
            )
            filter_parts.append(f"[delayed{i}]amix=inputs=2[mix{i}]")
            current_input = f"[mix{i}]"
        
        return ";".join(filter_parts) + f"{current_input}", input_files

    def combine_video_and_audio(self, muted_video: str, audio_files: List[Dict], 
                               output_video: str, video_duration: float) -> bool:
        """Combine muted video with generated commentary audio"""
        try:
            if not audio_files:
                print("No audio files to combine")
                return False
            
            # Create temporary combined audio file
            temp_audio = os.path.join(self.temp_dir, "combined_audio.wav")
            
            # Simple approach: overlay all audio files
            cmd = ['ffmpeg', '-f', 'lavfi', '-i', f'aevalsrc=0:duration={video_duration}:sample_rate=22050']
            
            # Add all audio inputs
            for audio_file in audio_files:
                cmd.extend(['-i', audio_file['audio_path']])
            
            # Create filter to mix all audio at proper timestamps
            filter_complex = "aevalsrc=0:duration={}:sample_rate=22050[base]".format(video_duration)
            
            for i, audio_file in enumerate(audio_files):
                delay_ms = int(audio_file['timestamp'] * 1000)
                filter_complex += f";[{i+1}:0]adelay={delay_ms}|{delay_ms}[delayed{i}]"
                if i == 0:
                    filter_complex += f";[base][delayed{i}]amix=inputs=2:duration=longest[mix{i}]"
                else:
                    filter_complex += f";[mix{i-1}][delayed{i}]amix=inputs=2:duration=longest[mix{i}]"
            
            final_output = f"[mix{len(audio_files)-1}]" if audio_files else "[base]"
            
            cmd.extend([
                '-filter_complex', filter_complex + final_output,
                '-y', temp_audio
            ])
            
            print("Creating combined audio track...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Audio mixing failed: {result.stderr}")
                return False
            
            # Combine video with audio
            cmd = [
                'ffmpeg', '-i', muted_video, '-i', temp_audio,
                '-c:v', 'copy', '-c:a', 'aac', '-shortest',
                '-y', output_video
            ]
            
            print("Combining video and audio...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            return result.returncode == 0
            
        except Exception as e:
            print(f"Error combining video and audio: {e}")
            return False

    def process_chess_video(self, input_video: str, output_video: str, 
                           president: str = "trump", frame_interval: int = 3, 
                           openai_api_key: str = None) -> bool:
        """Complete pipeline to process chess video with presidential commentary"""
        
        try:
            # Setup
            self.setup_temp_directory()
            
            print("🎬 Starting chess video processing pipeline...")
            print(f"Input: {input_video}")
            print(f"Output: {output_video}")
            print(f"President: {president}")
            
            # Step 1: Extract video info and validate
            video_info = self.extract_video_info(input_video)
            if not video_info:
                print("❌ Could not extract video information")
                return False
            
            print(f"📹 Video info: {video_info['duration']:.1f}s, {video_info['fps']:.1f}fps")
            
            # Step 2: Parse video frames
            frames, timestamps = self.parse_video_frames(input_video, frame_interval)
            if not frames:
                print("❌ No frames extracted")
                return False
            
            # Step 3: Analyze chess positions
            fens = self.analyze_chess_positions(frames, timestamps)
            if not fens:
                print("❌ No chess positions analyzed")
                return False
            
            # Step 4: Generate commentary
            events = self.generate_commentary(fens, timestamps, openai_api_key)
            if not events:
                print("❌ No commentary generated")
                return False
            
            # Step 5: Create audio files
            audio_dir = os.path.join(self.temp_dir, "audio")
            audio_files = self.create_audio_files(events, president, audio_dir)
            if not audio_files:
                print("❌ No audio files generated")
                return False
            
            # Step 6: Mute original video
            muted_video = os.path.join(self.temp_dir, "muted_video.mp4")
            if not self.mute_original_video(input_video, muted_video):
                print("❌ Failed to mute original video")
                return False
            
            # Step 7: Combine video and commentary
            if not self.combine_video_and_audio(muted_video, audio_files, 
                                               output_video, video_info['duration']):
                print("❌ Failed to combine video and audio")
                return False
            
            print("✅ Chess video processing completed successfully!")
            print(f"📁 Output saved to: {output_video}")
            
            # Generate summary
            self.print_processing_summary(events, audio_files)
            
            return True
            
        except Exception as e:
            print(f"❌ Error in video processing pipeline: {e}")
            return False
        
        finally:
            # Always cleanup
            self.cleanup_temp_directory()

    def print_processing_summary(self, events: List[CommentaryEvent], audio_files: List[Dict]):
        """Print a summary of the processing results"""
        print("\n📊 Processing Summary:")
        print(f"   • Commentary events: {len(events)}")
        print(f"   • Audio files generated: {len(audio_files)}")
        
        priority_counts = {}
        for event in events:
            priority_counts[event.priority] = priority_counts.get(event.priority, 0) + 1
        
        print(f"   • High priority comments: {priority_counts.get(3, 0)}")
        print(f"   • Medium priority comments: {priority_counts.get(2, 0)}")
        print(f"   • Low priority comments: {priority_counts.get(1, 0)}")
        
        if audio_files:
            total_duration = sum(len(af.get('audio_path', '')) for af in audio_files if af.get('audio_path'))
            print(f"   • Total commentary duration: ~{total_duration//60}min")
        
        print("\n🎯 Sample commentary:")
        for i, event in enumerate(events[:3]):  # Show first 3 events
            print(f"   {i+1}. [{event.timestamp:.1f}s] {event.text}")

# Convenience function for easy use
def process_chess_video_with_commentary(input_video: str, output_video: str, 
                                       president: str = "trump", 
                                       frame_interval: int = 3,
                                       openai_api_key: str = None) -> bool:
    """Easy-to-use function to process a chess video with presidential commentary"""
    processor = ChessVideoProcessor()
    return processor.process_chess_video(input_video, output_video, president, frame_interval, openai_api_key) 