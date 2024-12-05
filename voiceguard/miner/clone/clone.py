from voiceguard.protocol import VoiceGuardSynapse
# from TTS.api import TTS
import tempfile
import os

def voice_clone(self, synapse: VoiceGuardSynapse) -> bytes:
    # Initialize the YourTTS model
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", gpu=False)

    # Save the incoming reference audio (if in bytes) to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_ref_audio:
        temp_ref_audio.write(synapse.clone_clip)
        temp_ref_audio_path = temp_ref_audio.name

    # Define the text to synthesize
    text = synapse.clone_text

    # Define output path for the generated audio
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name

    try:
        # Generate the cloned voice
        tts.tts_to_file(text=text, speaker_wav=temp_ref_audio_path, language="en", file_path=output_path)

        print(f"Cloned voice generated and saved to: {output_path}")

        # Read the generated audio back into memory as bytes
        with open(output_path, "rb") as cloned_voice_file:
            cloned_voice = cloned_voice_file.read()

    finally:
        # Clean up temporary files
        os.remove(temp_ref_audio_path)
        os.remove(output_path)

    return cloned_voice