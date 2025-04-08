import torch
import soundfile as sf
import os
from rvc_python.infer import RVCInference

'''
Come back to this file later, maybe just proceed with tts instead of using rvc
'''

def tts():
    # base_dir = os.getcwd()

    # model_path = os.path.join(base_dir, "rvc_models", 'biden', "biden.pth")
    # model_path = os.path.join(base_dir, "rvc_models", 'trump', "trump.pth")
    # model_path = os.path.join(base_dir, "rvc_models", 'obama', "obama.pth")

    # rvc = RVCInference(device="cpu:0")
    # rvc.load_model(model_path)


    # Step 3: Convert Input Text to Speech (using TTS system like gTTS or Tacotron)
    import edge_tts

    TEXT = '''
    I'm feelin' lonely (Lonely)
    Oh, I wish I'd find a lover that could hold me (Hold me)
    Now I'm crying in my room
    So skeptical of love (Say what you say, but I want it more)
    But still, I want it more, more, more
    '''
    VOICE = "en-US-GuyNeural"
    tts = edge_tts.Communicate(TEXT, VOICE)
    tts.save_sync('output_speech.mp3')

    # Step 4: Apply the Voice Conversion (RVC) to Change the Speech to Biden's Voice
    # Assuming `convert_voice` is a function that applies the voice conversion
    # You may need to adjust this depending on the specific library you're using

    # converted_audio = rvc.infer_file('output_speech.mp3', "biden.wav")

    # audio_data = converted_audio[0] if isinstance(converted_audio, tuple) else converted_audio

    # Save the converted audio as a .wav file using soundfile
    # sf.write("output_biden_voice.wav", audio_data, 22050) 