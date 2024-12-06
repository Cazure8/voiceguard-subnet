import tempfile
import numpy as np
from scipy.spatial.distance import cosine
from speechbrain.pretrained import SpeakerRecognition
from typing import List
from voiceguard.utils.helper import transcribe_with_whisper
from voiceguard.validator.stt_reward import overall_correctness_score

def get_clone_rewards(
    self,
    clip_audio_path: str,
    clone_text: str,
    responses: List,
) -> List[float]:
    """
    Evaluate miner responses for voice cloning and calculate rewards based on voice similarity
    and textual correctness of the cloned audio.

    Args:
        clip_audio_path (str): Path to the reference audio clip.
        responses (List): Responses from miners containing cloned audio and text.
        clone_text (str): The expected text to be cloned.
        time_limit (int): The timeout limit for the response.

    Returns:
        List[float]: Rewards for each miner based on their response quality.
    """
    # Initialize a pre-trained speaker recognition model for embeddings
    verification_model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

    # Extract embeddings for the reference audio
    reference_embedding = verification_model.encode_file(clip_audio_path).detach().numpy()

    rewards = []

    # Process each miner's response
    for response in responses:
        try:
            # Extract audio from the response
            cloned_audio = response["clone_audio"]  # Assuming the miner's audio is returned as bytes
            
            if not cloned_audio:
                rewards.append(0.0)
                continue
            
            # Save the cloned audio temporarily
            with tempfile.NamedTemporaryFile(suffix=".wav") as temp_audio_file:
                temp_audio_file.write(cloned_audio)
                temp_audio_file.flush()  # Make sure data is written to disk

                # Apply speech-to-text to verify the presence of clone_text in the voice
                transcription = transcribe_with_whisper(temp_audio_file.name)
                
                # Extract embeddings for the cloned audio
                cloned_embedding = verification_model.encode_file(temp_audio_file.name).detach().numpy()

                # Compute cosine similarity between embeddings
                similarity_score = 1 - cosine(reference_embedding, cloned_embedding)

                # Evaluate text correctness score
                text_correctness_score = overall_correctness_score(clone_text, transcription)

                # Combine scores into a final reward
                if text_correctness_score > 0.8:
                    final_score = similarity_score
                else:
                    final_score = 0.0
                
                rewards.append(final_score)

        except Exception as e:
            print(f"Error processing response: {e}")
            rewards.append(0.0)  # Give a zero reward for failed processing

    return rewards