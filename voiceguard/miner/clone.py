import os
import base64
from voiceguard.protocol import VoiceGuardSynapse

def voice_clone(self, synapse: VoiceGuardSynapse, save_directory="miner_cloned_voices") -> bytes:
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
        print(f"Reference audio saved to: {ref_audio_path}")
        print(f"Cloned voice saved to: {cloned_audio_path}")

        with open(ref_audio_path, "rb") as ref_voice_file:
            ref_audio = ref_voice_file.read()

    finally:
        # Clean up only temporary files (not the saved files in the directory)
        pass

    return base64.b64encode(ref_audio).decode("utf-8")