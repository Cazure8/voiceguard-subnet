# The MIT License (MIT)
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

import numpy as np
import Levenshtein
import spacy
import re
import torch
import langid 
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import AutoTokenizer, AutoModel

nlp = spacy.load("en_core_web_lg")

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
model = AutoModel.from_pretrained("bert-base-multilingual-cased")

supported_languages = {
    # SpaCy supported languages with model codes, preferring "lg" where available
    'en': 'en_core_web_lg',      # English
    'de': 'de_core_news_lg',     # German
    'fr': 'fr_core_news_lg',     # French
    'es': 'es_core_news_lg',     # Spanish
    'pt': 'pt_core_news_lg',     # Portuguese
    'it': 'it_core_news_lg',     # Italian
    'nl': 'nl_core_news_lg',     # Dutch
    'el': 'el_core_news_md',     # Greek 
    'nb': 'nb_core_news_md',     # Norwegian Bokmål 
    'da': 'da_core_news_md',     # Danish 
    'ja': 'ja_core_news_md',     # Japanese 
    'zh': 'zh_core_web_md',      # Chinese 
    'ru': 'ru_core_news_md',     # Russian 
    'pl': 'pl_core_news_md',     # Polish 
    'ca': 'ca_core_news_md',     # Catalan 
    'uk': 'uk_core_news_md',     # Ukrainian 

    # Languages without official SpaCy models, fallback to BERT for robust multi-language support
    'ar': 'bert-base-multilingual-cased',   # Arabic
    'hi': 'bert-base-multilingual-cased',   # Hindi
    'ko': 'bert-base-multilingual-cased',   # Korean
    'sv': 'bert-base-multilingual-cased',   # Swedish
    'tr': 'bert-base-multilingual-cased',   # Turkish
    'bg': 'bert-base-multilingual-cased',   # Bulgarian
    'fa': 'bert-base-multilingual-cased',   # Persian
    'he': 'bert-base-multilingual-cased',   # Hebrew
    'id': 'bert-base-multilingual-cased',   # Indonesian
    'th': 'bert-base-multilingual-cased',   # Thai
    'vi': 'bert-base-multilingual-cased',   # Vietnamese
    'cs': 'bert-base-multilingual-cased',   # Czech
    'fi': 'bert-base-multilingual-cased',   # Finnish
    'hu': 'bert-base-multilingual-cased',   # Hungarian
    'ro': 'bert-base-multilingual-cased'    # Romanian
}
    
def load_spacy_or_bert_model(lang_code):
    try:
        if lang_code in supported_languages and 'core' in supported_languages[lang_code]:
            # Load SpaCy model with vectors if available
            model_to_load = supported_languages[lang_code]
            if 'md' in model_to_load or 'lg' in model_to_load:
                return spacy.load(model_to_load)
            else:
                print(f"Warning: Loaded SpaCy model does not support vector similarity for {lang_code}. Falling back to BERT.")
        # Fallback to BERT if no suitable SpaCy model
        tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
        model = AutoModel.from_pretrained('bert-base-multilingual-cased')
        return tokenizer, model
    except Exception as e:
        print(f"Error loading model for language {lang_code}: {e}")
        return spacy.load("en_core_web_md")  # Default fallback
    
def reward(query: str, response: str, response_time: float, max_response_time: float) -> float:
    print("-------response transcript from miner----------")
    print(response)
    print("-----------------------------------------------")
    if not re.match(r"\d+\$\$_\s*", response):
        return 0.0
    
    cleaned_response = re.sub(r"^\d+\$\$_\s*", "", response)
    print("--------cleaned----------")
    print(cleaned_response)
    print("=========================")

    lang = langid.classify(query)[0] 
    if lang not in supported_languages:
        lang = 'en'

    nlp_or_bert = load_spacy_or_bert_model(lang)

    if isinstance(nlp_or_bert, tuple):
        # BERT Model and Tokenizer
        model, tokenizer = nlp_or_bert
        query_tokens = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
        response_tokens = tokenizer(cleaned_response, return_tensors="pt", padding=True, truncation=True)

        try:
            # Forward pass to BERT model
            query_output = model(**query_tokens)
            response_output = model(**response_tokens)

            # Compute mean embeddings
            query_embeddings = query_output.last_hidden_state.mean(dim=1)
            response_embeddings = response_output.last_hidden_state.mean(dim=1)

            # Cosine similarity using PyTorch
            cosine_similarity_score = torch.nn.functional.cosine_similarity(
                query_embeddings, response_embeddings, dim=1
            ).item()

            correctness_score = overall_correctness_score(
                query, cleaned_response, additional_similarity_score=cosine_similarity_score
            )
        except Exception as e:
            print(f"Error during BERT model execution: {e}")
            correctness_score = 0.0
    else:
        query_doc = nlp_or_bert(query)
        response_doc = nlp_or_bert(cleaned_response)
        correctness_score = overall_correctness_score(query_doc.text, response_doc.text)
        print("--------nlp_correctness_score---------------")
        print(correctness_score)
        print("----------------------------------------")

    normalized_speed_score = 1 - response_time / max_response_time
    
    # Apply sigmoid to speed score for normalization between 0 and 1
    speed_score = sigmoid(np.array([normalized_speed_score]), temperature=1.0, shift=0.5).item()
    
    correctness_weight = 0.7
    speed_weight = 0.3
    
    combined_score = (correctness_weight * correctness_score) + (speed_weight * speed_score)

    return combined_score

def sigmoid(x, temperature=5, shift=0.5):
    """
    Apply a sigmoid function to normalize scores.
    """
    return 1 / (1 + np.exp(-temperature * (x - shift)))

def get_stt_rewards(self, query: str, responses, time_limit) -> np.ndarray:
    default_high_process_time = time_limit 
    response_times = np.array([
        response.dendrite.process_time if response.dendrite.process_time is not None else default_high_process_time
        for response in responses
    ])
    
    max_response_time = np.max(response_times)
    rewards = np.array([
        reward(
            query, 
            resp.stt_transcription, 
            resp.dendrite.process_time if resp.dendrite.process_time is not None else default_high_process_time, 
            max_response_time
        ) for resp in responses
    ])

    return rewards

def levenshtein_similarity(original, response):
    return 1 - Levenshtein.distance(original, response) / max(len(original), len(response))

def word_overlap_score(original, response):
    original_words = set(original.split())
    response_words = set(response.split())
    return len(original_words & response_words) / len(original_words)

def semantic_similarity(original, response, lang):
    nlp = load_spacy_or_bert_model(lang)
    if isinstance(nlp, tuple):
        return 0.4
    else:
        original_doc = nlp(original)
        response_doc = nlp(response)
        return original_doc.similarity(response_doc)

def overall_correctness_score(original, response, weight_overlap=0.25, weight_similarity=0.25, weight_levenshtein=0.25, weight_bleu=0.25, additional_similarity_score=None):
    if response is None:
        return 0.0

    # Handle the case when response is a single string
    overlap_score = word_overlap_score(original, response)
    levenshtein_score = levenshtein_similarity(original, response)
    bleu_score = get_bleu_score(original, response)

    lang = langid.classify(original)[0]  # Classify language based on original text
    if lang not in supported_languages:
        lang = 'en'

    # Calculate semantic similarity score
    if additional_similarity_score is not None:
        similarity_score = additional_similarity_score
    else:
        similarity_score = semantic_similarity(original, response, lang)

    # Combine scores with specified weights
    overall_score = (
        weight_overlap * overlap_score +
        weight_similarity * similarity_score +
        weight_levenshtein * levenshtein_score +
        weight_bleu * bleu_score
    )
    return overall_score

def get_bleu_score(reference, candidate):
    reference_tokens = [token.text for token in nlp(reference)]
    candidate_tokens = [token.text for token in nlp(candidate)]
    
    smoothing_function = SmoothingFunction().method4
    return sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothing_function)
