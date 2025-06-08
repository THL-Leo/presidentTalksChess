#!/usr/bin/env python3
"""
Chess Video Commentary System with Presidential Voices
=====================================================

This script processes chess videos and adds presidential commentary using:
1. Chess piece detection with YOLO
2. Position analysis with Stockfish  
3. Natural language commentary generation
4. Presidential voice synthesis using Edge TTS + RVC
5. Video processing with ffmpeg

Usage:
    python src/main_commentary.py [video_path] [--president trump|biden|obama] [--interval 3]

Requirements:
    - Video file with visible chess board
    - Stockfish engine in ./stockfish/
    - RVC models in ./rvc_models/ (optional)
    - ffmpeg installed
"""

import argparse
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.video_processor import process_chess_video_with_commentary
from src.presidential_tts import PresidentialTTS
import asyncio

def check_dependencies():
    """Check if required dependencies are available"""
    print("🔍 Checking dependencies...")
    
    issues = []
    
    # Check Stockfish
    stockfish_path = "./stockfish/stockfish-macos-m1-apple-silicon"
    if not os.path.exists(stockfish_path):
        issues.append(f"❌ Stockfish not found at {stockfish_path}")
    else:
        print("✅ Stockfish engine found")
    
    # Check ffmpeg
    import subprocess
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        print("✅ ffmpeg found")
    except (subprocess.CalledProcessError, FileNotFoundError):
        issues.append("❌ ffmpeg not found - please install ffmpeg")
    
    # Check RVC models (optional)
    rvc_dir = "rvc_models"
    if os.path.exists(rvc_dir):
        presidents = ["trump", "biden", "obama"]
        found_models = []
        for president in presidents:
            model_path = os.path.join(rvc_dir, president, f"{president}.pth")
            if os.path.exists(model_path):
                found_models.append(president)
        
        if found_models:
            print(f"✅ RVC models found for: {', '.join(found_models)}")
        else:
            print("⚠️  No RVC models found - will use base TTS voices")
    else:
        print("⚠️  RVC models directory not found - will use base TTS voices")
    
    # Check YOLO model
    yolo_path = "model/best.pt"
    if not os.path.exists(yolo_path):
        issues.append(f"❌ YOLO model not found at {yolo_path}")
    else:
        print("✅ YOLO chess detection model found")
    
    if issues:
        print("\n🚨 Issues found:")
        for issue in issues:
            print(f"   {issue}")
        print("\n💡 Please resolve these issues before running the system.")
        return False
    
    print("✅ All dependencies check passed!")
    return True

async def test_voice_generation(president: str = "trump"):
    """Test the voice generation system"""
    print(f"\n🎤 Testing {president} voice generation...")
    
    tts = PresidentialTTS()
    success = await tts.generate_sample_audio(president)
    
    if success:
        print("✅ Voice generation test successful!")
        print(f"📁 Sample audio saved as: sample_{president}_voice.wav")
    else:
        print("❌ Voice generation test failed")
    
    return success

def main():
    parser = argparse.ArgumentParser(
        description="Generate presidential chess commentary for videos",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "video_path", 
        nargs='?',
        help="Path to the chess video file"
    )
    
    parser.add_argument(
        "--president", "-p",
        choices=["trump", "biden", "obama"],
        default=os.getenv("DEFAULT_PRESIDENT", "trump"),
        help="Presidential voice to use (default: trump or env DEFAULT_PRESIDENT)"
    )
    
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=int(os.getenv("DEFAULT_FRAME_INTERVAL", "3")),
        help="Frame extraction interval in seconds (default: 3 or env DEFAULT_FRAME_INTERVAL)"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output video path (default: adds '_commentary' to input filename)"
    )
    
    parser.add_argument(
        "--test-voice",
        action="store_true",
        help="Test voice generation without processing video"
    )
    
    parser.add_argument(
        "--check-deps",
        action="store_true", 
        help="Check dependencies and exit"
    )
    
    parser.add_argument(
        "--openai-key",
        help="OpenAI API key for GPT commentary generation"
    )
    
    args = parser.parse_args()
    
    print("🎬 Chess Video Commentary System")
    print("=" * 50)
    
    # Check dependencies
    if args.check_deps or not check_dependencies():
        return 1 if args.check_deps else 1
    
    # Test voice generation
    if args.test_voice:
        success = asyncio.run(test_voice_generation(args.president))
        return 0 if success else 1
    
    # Validate video input
    if not args.video_path:
        print("❌ No video path provided")
        print("Usage: python src/main_commentary.py <video_path>")
        print("   or: python src/main_commentary.py --test-voice")
        return 1
    
    if not os.path.exists(args.video_path):
        print(f"❌ Video file not found: {args.video_path}")
        return 1
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        base_name = os.path.splitext(args.video_path)[0]
        output_path = f"{base_name}_commentary_{args.president}.mp4"
    
    print(f"📹 Input video: {args.video_path}")
    print(f"🎭 President: {args.president}")
    print(f"⏱️  Frame interval: {args.interval}s")
    print(f"📁 Output: {output_path}")
    
    # Process the video
    print("\n🚀 Starting video processing...")
    
    success = process_chess_video_with_commentary(
        input_video=args.video_path,
        output_video=output_path,
        president=args.president,
        frame_interval=args.interval,
        openai_api_key=args.openai_key
    )
    
    if success:
        print(f"\n🎉 Success! Commentary video created: {output_path}")
        
        # Show some usage tips
        print("\n💡 Tips:")
        print("   • Try different presidents with --president flag")
        print("   • Adjust timing with --interval flag (smaller = more frequent commentary)")
        print("   • Place RVC models in ./rvc_models/[president]/ for better voice quality")
        
        return 0
    else:
        print("\n❌ Video processing failed")
        print("\n🔧 Troubleshooting:")
        print("   • Check that the video shows a clear chess board")
        print("   • Ensure all dependencies are installed")
        print("   • Try with --check-deps to verify setup")
        print("   • Use --test-voice to test audio generation")
        
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⏹️  Process interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1) 