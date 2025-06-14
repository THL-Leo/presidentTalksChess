import os
import asyncio
import logging
import edge_tts
import torch
import soundfile as sf
from rvc_python.infer import RVCInference
from typing import List, Dict, Optional
import tempfile
from src.chess_commentator import CommentaryEvent

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class PresidentialTTS:
    def __init__(self, rvc_models_dir: str = "rvc_models"):
        self.rvc_models_dir = rvc_models_dir
        self.rvc_inference = None
        self.device = "mps"  # Change to "cuda" if you have GPU
        
        # Voice mappings for Edge TTS
        self.base_voices = {
            "male_1": "en-US-GuyNeural"
        }
        
        # Available presidential voices (add as you get more RVC models)
        self.presidential_voices = {
            "trump": {
                "model_path": os.path.join(rvc_models_dir, "trump", "trump.pth"),
                "base_voice": "male_1",
                "personality": "confident"
            },
            "biden": {
                "model_path": os.path.join(rvc_models_dir, "biden", "biden.pth"),
                "base_voice": "male_1", 
                "personality": "measured"
            },
            "obama": {
                "model_path": os.path.join(rvc_models_dir, "obama", "obama.pth"),
                "base_voice": "male_1",
                "personality": "smooth"
            }
        }
        
        # Commentary style adjustments based on personality
        self.personality_adjustments = {
            "confident": {
                "prefix_options": ["Let me tell you,", "Believe me,", "This is huge,", "Fantastic move!"],
                "suffix_options": ["Tremendous!", "The best!", "Incredible!", "Perfect!"]
            },
            "measured": {
                "prefix_options": ["Well,", "You know,", "Here's the thing,", "Let me be clear,"],
                "suffix_options": ["That's the truth.", "Period.", "Come on.", "Seriously."]
            },
            "smooth": {
                "prefix_options": ["Now,", "Let me be clear,", "Here's what I see,", "The fact is,"],
                "suffix_options": ["That's right.", "Indeed.", "Absolutely.", "No doubt about it."]
            }
        }

    def initialize_rvc(self):
        """Initialize RVC inference engine"""
        try:
            self.rvc_inference = RVCInference(device=self.device)
            print("RVC inference engine initialized")
            return True
        except Exception as e:
            print(f"Failed to initialize RVC: {e}")
            logger.exception("RVC initialization error details:")
            return False

    async def generate_base_speech(self, text: str, voice: str, output_path: str):
        """Generate speech using Edge TTS"""
        try:
            logger.debug(f"Generating TTS: voice={voice}, text='{text[:50]}...', output={output_path}")
            tts = edge_tts.Communicate(text, voice)
            await tts.save(output_path)
            
            # Verify file was created
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                logger.debug(f"TTS generated successfully: {file_size} bytes")
                return True
            else:
                logger.error(f"TTS file was not created: {output_path}")
                return False
        except Exception as e:
            logger.error(f"Error generating base speech: {e}")
            print(f"Error generating base speech: {e}")
            return False

    def convert_voice_with_rvc(self, input_audio: str, model_path: str, output_path: str) -> bool:
        """Convert voice using RVC model"""
        try:
            if not self.rvc_inference:
                if not self.initialize_rvc():
                    return False
            
            # Load the specific RVC model
            self.rvc_inference.load_model(model_path)
            
            # Perform voice conversion
            self.rvc_inference.infer_file(input_audio, output_path)
            
        except Exception as e:
            print(f"Error in RVC conversion: {e}")
            logger.exception("RVC conversion error details:")
            return False

    def adjust_text_for_personality(self, text: str, personality: str) -> str:
        """Adjust commentary text based on presidential personality"""
        if personality not in self.personality_adjustments:
            return text
        
        adjustments = self.personality_adjustments[personality]
        
        # Sometimes add personality-specific prefixes/suffixes
        import random
        if random.random() < 0.3:  # 30% chance to add prefix
            prefix = random.choice(adjustments["prefix_options"])
            text = f"{prefix} {text}"
        
        if random.random() < 0.2:  # 20% chance to add suffix
            suffix = random.choice(adjustments["suffix_options"])
            text = f"{text} {suffix}"
        
        return text

    async def generate_presidential_audio(self, text: str, president: str, output_path: str) -> bool:
        """Generate audio with presidential voice"""
        if president not in self.presidential_voices:
            print(f"President '{president}' not available. Available: {list(self.presidential_voices.keys())}")
            return False
        
        voice_config = self.presidential_voices[president]
        model_path = voice_config["model_path"]
        base_voice = self.base_voices[voice_config["base_voice"]]
        personality = voice_config["personality"]
        
        # Check if RVC model exists
        if not os.path.exists(model_path):
            print(f"RVC model not found: {model_path}")
            print("Falling back to base TTS voice...")
            # Just use base TTS without RVC conversion
            adjusted_text = self.adjust_text_for_personality(text, personality)
            return await self.generate_base_speech(adjusted_text, base_voice, output_path)
        
        # Adjust text for personality
        adjusted_text = self.adjust_text_for_personality(text, personality)
        
        # Generate base speech with temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Generate base speech
            if not await self.generate_base_speech(adjusted_text, base_voice, temp_path):
                return False
            
            # Convert with RVC
            success = self.convert_voice_with_rvc(temp_path, model_path, output_path)
            
            if not success:
                print(f"RVC conversion failed, falling back to base TTS for {president}")
                # Copy the base TTS file as fallback
                import shutil
                try:
                    shutil.copy2(temp_path, output_path)
                    print(f"Fallback: copied base TTS to {output_path}")
                    return True
                except Exception as copy_error:
                    print(f"Failed to copy fallback audio: {copy_error}")
                    return False
            
            return success
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    async def process_commentary_events(self, events: List[CommentaryEvent], 
                                      president: str, output_dir: str) -> List[Dict]:
        """Process all commentary events and generate audio files"""
        logger.info(f"Processing {len(events)} commentary events for {president}")
        os.makedirs(output_dir, exist_ok=True)
        
        audio_files = []
        
        for i, event in enumerate(events):
            logger.debug(f"Processing event {i+1}/{len(events)}: timestamp={event.timestamp:.1f}s, text='{event.text[:50]}...'")
            print(f"Processing commentary {i+1}/{len(events)}: {event.text[:50]}...")
            
            # Generate filename
            audio_filename = f"commentary_{i:03d}_{event.timestamp:.1f}s.wav"
            audio_path = os.path.join(output_dir, audio_filename)
            
            # Generate presidential audio
            success = await self.generate_presidential_audio(
                event.text, president, audio_path
            )
            
            if success:
                audio_files.append({
                    "timestamp": event.timestamp,
                    "audio_path": audio_path,
                    "text": event.text,
                    "priority": event.priority,
                    "move_san": event.move_san
                })
                logger.info(f"✓ Generated audio file: {audio_filename}")
                print(f"✓ Generated: {audio_filename}")
            else:
                logger.error(f"✗ Failed to generate audio: {audio_filename}")
                print(f"✗ Failed to generate: {audio_filename}")
        
        logger.info(f"Audio generation complete: {len(audio_files)}/{len(events)} files created")
        return audio_files

    async def generate_sample_audio(self, president: str = "trump"):
        """Generate a sample audio to test the system"""
        sample_text = "What a fantastic chess move! This is tremendous tactical play, believe me!"
        output_path = f"sample_{president}_voice.wav"
        
        success = await self.generate_presidential_audio(sample_text, president, output_path)
        
        if success:
            print(f"Sample audio generated: {output_path}")
        else:
            print("Failed to generate sample audio")
        
        return success

# Async helper function for easy use
async def generate_presidential_commentary(events: List[CommentaryEvent], 
                                         president: str = "trump",
                                         output_dir: str = "audio_output") -> List[Dict]:
    """Helper function to generate all presidential commentary audio"""
    tts = PresidentialTTS()
    return await tts.process_commentary_events(events, president, output_dir)

# Sync wrapper for the async function
def generate_presidential_commentary_sync(events: List[CommentaryEvent], 
                                        president: str = "trump",
                                        output_dir: str = "audio_output") -> List[Dict]:
    """Synchronous wrapper for generating presidential commentary"""
    return asyncio.run(generate_presidential_commentary(events, president, output_dir)) 