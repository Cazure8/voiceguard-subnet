import numpy as np
from typing import List

def get_detection_rewards(
    self,
    detection_audio: bytes,
    responses: List,
    ground_truth: bool,
    time_limit: int,
) -> List[float]:
    """
    Evaluate miner responses for voice deepfake detection and calculate rewards.

    Args:
        self: The object containing the state and necessary configurations.
        detection_audio (bytes): The audio file to be checked for deepfake as bytes.
        responses (List): Responses from miners containing confidence percentages.
        ground_truth (bool): The actual label of the audio (True = real, False = fake).
        time_limit (int): The timeout limit for the response.

    Returns:
        List[float]: Rewards for each miner based on their response quality.
    """
    rewards = []

    for response in responses:
        try:
            # Extract miner's confidence percentage (real vs fake)
            miner_confidence = response.get("confidence")  # A float value between 0.0 and 1.0
            miner_predicted_label = miner_confidence >= 0.5  # True = real, False = fake

            # Calculate correctness score
            correctness_score = 1.0 if miner_predicted_label == ground_truth else 0.0

            # Calculate confidence adjustment
            confidence_adjustment = miner_confidence if ground_truth else (1 - miner_confidence)

            # Combine correctness and confidence for the final reward
            reward = correctness_score * confidence_adjustment
            rewards.append(reward)

        except Exception as e:
            print(f"Error processing response: {e}")
            rewards.append(0.0)  # Assign zero reward for any errors

    return rewards