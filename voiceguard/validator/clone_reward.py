import numpy as np
from scipy.spatial.distance import cosine
from speechbrain.pretrained import SpeakerRecognition
import tempfile
import os
from typing import List

def get_clone_rewards(
    self,
    clone_clip: bytes,
    responses: List,
    clone_text: str,
    time_limit: int,
) -> List[float]:
    """
    Evaluate miner responses for voice cloning and calculate rewards.

    Args:
        self: The object containing the state and necessary configurations.
        clone_clip (bytes): The reference audio clip as bytes.
        responses (List): Responses from miners containing cloned audio and text.
        clone_text (str): The expected text to be cloned.
        time_limit (int): The timeout limit for the response.

    Returns:
        List[float]: Rewards for each miner based on their response quality.
    """
    # Initialize a pre-trained speaker recognition model for embeddings
    verification_model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

    # Save the reference audio temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_ref_audio:
        temp_ref_audio.write(clone_clip)
        temp_ref_audio_path = temp_ref_audio.name

    # Extract embeddings for the reference audio
    reference_embedding = verification_model.encode_file(temp_ref_audio_path).detach().numpy()

    rewards = []

    try:
        # Process each miner's response
        for response in responses:
            try:
                # Extract audio and text from the response
                cloned_audio = response["audio"]  # Assuming the miner's audio is returned as bytes
                cloned_text = response["text"]    # Assuming the miner's text is returned

                # Save the cloned audio temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
                    temp_audio_file.write(cloned_audio)
                    cloned_audio_path = temp_audio_file.name

                # Extract embeddings for the cloned audio
                cloned_embedding = verification_model.encode_file(cloned_audio_path).detach().numpy()

                # Compute cosine similarity between embeddings
                similarity_score = 1 - cosine(reference_embedding, cloned_embedding)

                # Text correctness score
                text_correctness_score = 1.0 if cloned_text.strip() == clone_text.strip() else 0.0

                # Combine scores into a final reward (weight similarity and text correctness equally)
                final_score = (similarity_score + text_correctness_score) / 2.0
                rewards.append(final_score)

                # Cleanup temporary audio file
                os.remove(cloned_audio_path)

            except Exception as e:
                print(f"Error processing response: {e}")
                rewards.append(0.0)  # Give a zero reward for failed processing

    finally:
        # Clean up reference audio file
        os.remove(temp_ref_audio_path)

    return rewards