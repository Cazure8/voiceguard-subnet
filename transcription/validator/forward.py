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

import os
import time
import random
import io
import bittensor as bt
import base64
import glob
import pyttsx3
import torchaudio
from pydub import AudioSegment
from requests.exceptions import HTTPError
from transcription.protocol import Transcription
from transcription.validator.reward import get_rewards
from transcription.utils.uids import get_random_uids
from gtts import gTTS, gTTSError

async def forward(self):
    """
    The forward function is called by the validator every time step.

    It is responsible for querying the network and scoring the responses.

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

    """
    audio_sample, ground_truth_transcription = generate_or_load_audio_sample()
    audio_sample_base64 = encode_audio_to_base64(audio_sample)
    miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)
    
    # The dendrite client queries the network.
    responses = self.dendrite.query(
        # Send the query to selected miner axons in the network.
        axons=[self.metagraph.axons[uid] for uid in miner_uids],
        synapse=Transcription(audio_input=audio_sample_base64),
        deserialize=True,
    )

    # Log the results for monitoring purposes.
    bt.logging.info(f"Received responses: {responses}")

    # Adjust the scores based on responses from miners.
    rewards = get_rewards(self, query=ground_truth_transcription, responses=responses)

    bt.logging.info(f"Scored responses: {rewards}")
    # Update the scores based on the rewards. You may want to define your own update_scores function for custom behavior.
    self.update_scores(rewards, miner_uids)
    
def generate_or_load_audio_sample(base_path='librispeech_dataset'):
    # Attempt to generate audio using TTS
    script = generate_random_text(num_sentences=25, sentence_length=10)
    mp3_file_name = f"temp_{random.randint(0, 10000)}.mp3"

    if random.random() < 0.6 and (google_tts(script, mp3_file_name) or local_tts(script, mp3_file_name)):
        audio_data_flac, script = convert_mp3_to_flac(mp3_file_name, script)
        if audio_data_flac:
            return audio_data_flac, script

    # If TTS fails or is not used, try loading from LibriSpeech dataset
    audio_data, transcript = load_from_librispeech(base_path)
    if audio_data:
        return audio_data, transcript

    # As a last resort, search through the entire dataset to find any valid audio file and its transcript
    return search_for_any_audio_file(base_path)

def convert_mp3_to_flac(mp3_file_name, script):
    try:
        audio_file_flac = mp3_file_name.replace('.mp3', '.flac')
        sound = AudioSegment.from_mp3(mp3_file_name)
        sound.export(audio_file_flac, format="flac")
        os.remove(mp3_file_name)  # Clean up the generated MP3 file
        with open(audio_file_flac, 'rb') as file:
            audio_data_flac = file.read()
        os.remove(audio_file_flac)  # Clean up the generated FLAC file
        return audio_data_flac, script
    except Exception as e:
        print(f"Error during MP3 to FLAC conversion: {e}")
        return None, None
    
def load_from_librispeech(base_path):
    subsets = ['train-clean-100', 'train-clean-360', 'train-other-500', 'dev-clean', 'dev-other', 'test-clean', 'test-other']
    for subset in subsets:
        subset_path = os.path.join(base_path, 'LibriSpeech', subset)
        for audio_data, transcript in iterate_audio_files(subset_path):
            if audio_data:
                return audio_data, transcript
    return None, None

def iterate_audio_files(subset_path):
    for speaker_dir in glob.glob(os.path.join(subset_path, '*/')):
        for chapter_dir in glob.glob(os.path.join(speaker_dir, '*/')):
            transcript_file = glob.glob(os.path.join(chapter_dir, "*.trans.txt"))[0]
            with open(transcript_file, 'r') as file:
                for line in file:
                    audio_filename, transcript = line.strip().split(' ', 1)
                    audio_filepath = os.path.join(chapter_dir, audio_filename + '.flac')
                    if os.path.exists(audio_filepath):
                        with open(audio_filepath, 'rb') as audio_file:
                            return audio_file.read(), transcript
    return None, None

def search_for_any_audio_file(base_path):
    """Searches through the entire dataset to find and return any audio file and its transcript."""
    subsets = ['train-clean-100', 'train-clean-360', 'train-other-500', 'dev-clean', 'dev-other', 'test-clean', 'test-other']
    for subset in subsets:
        subset_path = os.path.join(base_path, 'LibriSpeech', subset)
        audio_data, transcript = iterate_audio_files(subset_path)
        if audio_data:
            return audio_data, transcript
    # This point should theoretically never be reached if there are always files available
    raise FileNotFoundError("No audio files were found in the dataset.")

def google_tts(script, filename):
    try:
        tts = gTTS(script)
        tts.save(filename)
        return True
    except HTTPError as e:
        if e.response.status_code == 429:
            print("Hit rate limit for Google TTS")
            return False
        elif e.response.status_code == 500:
            print("Internal Server Error from TTS API")
            return False
        elif e.response.status_code == 503:
            print("Service Unavailable. TTS API might be down or undergoing maintenance")
            return False
        elif e.response.status_code == 401:
            print("Unauthorized. Check your API key or authentication method")
            return False
        elif e.response.status_code == 403:
            print("Forbidden. You might not have permission to use this service")
            return False
        raise
    except gTTSError as e:
        if "429 (Too Many Requests)" in str(e):
            print("Hit rate limit for Google TTS")
            return False
        elif "500 (Internal Server Error)" in str(e):
            print("Internal Server Error from TTS API. Probably cause: Upstream API error")
            return False
        elif "Failed to connect" in str(e):
            print("Connection error in Google TTS")
            return False
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return False

def local_tts(script, filename):
    engine = pyttsx3.init()
    engine.save_to_file(script, filename)
    engine.runAndWait()
    # Wait for the file to be created
    timeout = 20  # Maximum number of seconds to wait
    start_time = time.time()
    while not os.path.exists(filename) or os.path.getsize(filename) == 0:
        time.sleep(1)
        if time.time() - start_time > timeout:
            print("Timeout waiting for TTS file to be created.")
            return False
    time.sleep(2)
    
    return True

def waveform_to_binary(waveform, sample_rate):
    """
    Converts a waveform tensor back to binary audio data.
    """
    binary_stream = io.BytesIO()
    torchaudio.save(binary_stream, waveform, sample_rate, format="wav")
    binary_stream.seek(0)
    return binary_stream.read()

def generate_random_sentence(words, length=10):
    return ' '.join(random.choices(words, k=length)) + '.'

def generate_random_text(num_sentences=5, sentence_length=5):
    common_words = ['the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'I', 
                    'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at', 
                    'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 
                    'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 
                    'their', 'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 
                    'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time', 'no', 
                    'just', 'him', 'know', 'take', 'person', 'into', 'year', 'your', 
                    'good', 'some', 'could', 'them', 'see', 'other', 'than', 'then', 
                    'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also', 
                    'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 
                    'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these', 
                    'give', 'day', 'most', 'us']

    return ' '.join([generate_random_sentence(common_words, sentence_length) for _ in range(num_sentences)])

def encode_audio_to_base64(audio_data):
    # Encode binary audio data to Base64 string
    return base64.b64encode(audio_data).decode('utf-8')
