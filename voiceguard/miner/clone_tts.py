import os
import base64
from voiceguard.protocol import VoiceGuardSynapse
from TTS.api import TTS


def voice_clone(self, synapse: VoiceGuardSynapse, save_directory="miner_cloned_voices") -> bytes:
    # Initialize the YourTTS model
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", gpu=False)

    # Decode the Base64 audio clip into bytes
    clone_clip_bytes = base64.b64decode(synapse.clone_clip)

    # Ensure the save directory exists
    os.makedirs(save_directory, exist_ok=True)

    # Generate a unique identifier for the files (e.g., request ID or timestamp)
    file_id = f"{len(os.listdir(save_directory)) + 1}"

    # Define file paths for the reference and cloned audio
    ref_audio_path = os.path.join(save_directory, f"{file_id}_clip.wav")
    cloned_audio_path = os.path.join(save_directory, f"{file_id}_cloned.wav")

    # Save the reference audio clip
    with open(ref_audio_path, "wb") as ref_audio_file:
        ref_audio_file.write(clone_clip_bytes)

    try:
        # Generate the cloned voice
        tts.tts_to_file(text=synapse.clone_text, speaker_wav=ref_audio_path, language="en", file_path=cloned_audio_path)

        print(f"Reference audio saved to: {ref_audio_path}")
        print(f"Cloned voice saved to: {cloned_audio_path}")

        # Read the generated cloned audio back into memory as bytes
        with open(cloned_audio_path, "rb") as cloned_voice_file:
            cloned_voice = cloned_voice_file.read()

    finally:
        # Clean up only temporary files (not the saved files in the directory)
        pass

    return base64.b64encode(cloned_voice).decode("utf-8")