import numpy as np
from typing import List
from sklearn.metrics import f1_score, accuracy_score

def get_detection_rewards(
    self,
    ground_truth: bool,
    responses: List,
    time_limit: int,
) -> List[float]:
    """
    Evaluate miner responses for voice deepfake detection and calculate rewards based on
    both the accuracy and the confidence of the responses.

    Args:
        ground_truth (bool): The actual label of the audio (True if real, False if fake).
        responses (List): Miner responses, each containing a confidence level of the prediction.
        time_limit (int): The maximum allowed time for responses.

    Returns:
        List[float]: Rewards for each miner based on their response quality.
    """
    rewards = []

    # Initialize lists to store binary predictions and confidences
    predictions = []
    confidences = []

    # Collect predictions and their confidences
    for response in responses:
        miner_confidence = response.get("detection_prediction")  # Assuming this is a probability [0,1]
        miner_predicted_label = miner_confidence >= 0.5  # True if real, False if fake
        predictions.append(miner_predicted_label)
        confidences.append(miner_confidence)

    # Convert lists to numpy arrays for metric calculations
    predictions = np.array(predictions)
    confidences = np.array(confidences)
    ground_truth_array = np.array([ground_truth] * len(predictions))  # Same truth value repeated for comparison

    # Calculate binary metrics
    accuracy = accuracy_score(ground_truth_array, predictions)
    f1 = f1_score(ground_truth_array, predictions)

    for idx, (prediction, confidence) in enumerate(zip(predictions, confidences)):
        # Calculate correctness score
        correctness_score = 1.0 if prediction == ground_truth else 0.0

        # Adjust reward based on confidence
        confidence_adjustment = confidence if ground_truth else (1 - confidence)
        
        # Combine accuracy, f1, and adjusted confidence for final reward
        reward = (correctness_score * confidence_adjustment * accuracy * f1)
        rewards.append(reward)

    return rewards