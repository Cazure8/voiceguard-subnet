# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 Cazure

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import torch
from typing import List, Optional, Union
import Levenshtein
import spacy

nlp = spacy.load("en_core_web_md")

def reward(query: str, response: str) -> float:
    """
    Reward the miner response to the dummy request. This method returns a reward
    value for the miner, which is used to update the miner's score.

    Returns:
    - float: The reward value for the miner.
    """
    correctness_scores = overall_correctness_score(query, response)
    # speed_scores = score_response_speed(response_times, responses_casted)
    
    
    # combined_scores = []
    # for correctness, speed in zip(correctness_scores, speed_scores):
    #     combined_score = 0.7 * correctness + 0.3 * speed
    #     combined_scores.append(combined_score)
    return correctness_scores


def get_rewards(
    self,
    query: str,
    responses: List[float],
) -> torch.FloatTensor:
    """
    Returns a tensor of rewards for the given query and responses.

    Args:
    - query (int): The query sent to the miner.
    - responses (List[float]): A list of responses from the miner.

    Returns:
    - torch.FloatTensor: A tensor of rewards for the given query and responses.
    """
    # Get all the reward results by iteratively calling your reward() function.
    return torch.FloatTensor(
        [reward(query, response) for response in responses]
    ).to(self.device)

def levenshtein_similarity(original, response):
    distance = Levenshtein.distance(original, response)
    max_length = max(len(original), len(response))
    similarity_score = (max_length - distance) / max_length
    return similarity_score

def word_overlap_score(original, response):
    original_words = set(original.split())
    response_words = set(response.split())
    common_words = original_words.intersection(response_words)
    overlap_score = len(common_words) / len(original_words)
    return overlap_score

def semantic_similarity(original, response):
    original_doc = nlp(original)
    response_doc = nlp(response)
    similarity_score = original_doc.similarity(response_doc)
    return similarity_score

def overall_correctness_score(original, response, weight_overlap=0.4, weight_similarity=0.4, weight_levenshtein=0.2):
    if response is None:
        return 0.0
    # Handle the case when response is a single string
    overlap_score = word_overlap_score(original, response)
    similarity_score = semantic_similarity(original, response)
    levenshtein_score = levenshtein_similarity(original, response)
    
    # Combine scores with specified weights
    overall_score = (
        weight_overlap * overlap_score +
        weight_similarity * similarity_score +
        weight_levenshtein * levenshtein_score
    )
    return overall_score



def score_response_speed(response_times: List[float], responses: Optional[Union[List[Optional[str]], List[None]]]) -> List[float]:
    """
    Scores the response speed based on response times and the existence of a response.

    Args:
    - response_times (List[float]): List of response times for each miner.
    - responses (Optional[Union[List[Optional[str]], List[None]]]): An optional list of responses from the miner, where each response is either a string, None, or the entire list can be None.

    Returns:
    - List[float]: A list of scores for the response speed, considering the presence of responses.
    """
    # Ensure that responses is a list of the appropriate length, even if it's empty or None
    if responses is None:
        responses = [None] * len(response_times)

    # Calculate the maximum response time for normalization
    max_time = max(response_times, default=0)

    speed_scores = []
    for time, response in zip(response_times, responses):
        # Assign a score of 0 if the response is None, regardless of the speed
        if response is None:
            speed_scores.append(0.0)
        else:
            # Score based on response time (faster responses get higher scores)
            # Normalized against the maximum response time
            speed_scores.append(1 - (time / max_time)) if max_time > 0 else 0.0

    return speed_scores