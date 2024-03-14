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
from typing import List
import Levenshtein
import spacy
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize

nlp = spacy.load("en_core_web_md")
nltk.download('punkt')

def reward(query: str, response: str, response_time: float, max_response_time: float, type: str) -> float:
    if response is None or response.strip() == "":
        correctness_score = 0.0 
        speed_score = 0.0 
    else:
        if type == "url":
            correctness_score = 1 if get_bleu_score(query, response) > 0.3 else 0
        else:
            correctness_score = overall_correctness_score(query, response)
            
        
        normalized_speed_score = 1 - response_time / max_response_time
        
        # Apply sigmoid to speed score for normalization between 0 and 1
        speed_score = sigmoid(torch.tensor([normalized_speed_score]), temperature=1.0, shift=0.5).item()
        
    correctness_weight = 0.6
    speed_weight = 0.4
    
    combined_score = (correctness_weight * correctness_score) + (speed_weight * speed_score)

    return combined_score

def sigmoid(x, temperature=1.0, shift=0.0):
    """
    Apply a sigmoid function to normalize scores.
    """
    return 1 / (1 + torch.exp(-temperature * (x - shift)))

def get_rewards(self, query: str, responses, type: str, time_limit) -> torch.FloatTensor:
    print("-----here is getting reward part------")
    default_high_process_time = time_limit 
    response_times = torch.FloatTensor([
        response.dendrite.process_time if response.dendrite.process_time is not None else default_high_process_time
        for response in responses
    ])
    
    max_response_time = torch.max(response_times)
    rewards = torch.FloatTensor([
    reward(
        query, 
        resp.transcription_output, 
        resp.dendrite.process_time if resp.dendrite.process_time is not None else default_high_process_time, 
        max_response_time.item(),
        type
        ) for resp in responses
    ])

    return rewards.to(self.device)

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
    if response is None or response.strip() == "":
        return 0.0 
    
    original_doc = nlp(original)
    response_doc = nlp(response)
    
    if not original_doc.has_vector or not response_doc.has_vector:
        return 0.4

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

def get_bleu_score(request, response):
    # Tokenize the reference and candidate sentences
    request_tokens = [word_tokenize(request.lower())]
    response_tokens = word_tokenize(response.lower())
    
    weights = (1, 0, 0, 0)  # This gives full weight to 1-gram precision, ignoring longer n-grams
    score = sentence_bleu(request_tokens, response_tokens, weights=weights)
    
    return score