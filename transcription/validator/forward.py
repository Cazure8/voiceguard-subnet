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
from gtts import gTTS

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
    # Option 1: Generate an audio file from a script
    if random.random() < 0.6:  # 60% chance to use TTS
        script = generate_random_text(num_sentences=30, sentence_length=10)
        mp3_file_name = "temp_{}.mp3".format(random.randint(0, 10000))

        if google_tts(script, mp3_file_name):
            bt.logging.info("Using Google TTS")
        else:
            bt.logging.info("Using local TTS")
            if not local_tts(script, mp3_file_name):
                return None, None 
            
        audio_file_flac = mp3_file_name.replace('.mp3', '.flac')
        try:
            sound = AudioSegment.from_mp3(mp3_file_name)
            sound_16k = sound.set_frame_rate(16000)
            sound_16k.export(audio_file_flac, format="flac")
            os.remove(mp3_file_name)  # Clean up the generated MP3 file
        except Exception as e:
            print(f"Error converting MP3 to FLAC: {e}")
            return None, None

        try:
            with open(audio_file_flac, 'rb') as file:
                audio_data_flac = file.read()
            os.remove(audio_file_flac)  # Clean up the generated FLAC file
        except Exception as e:
            print(f"Error handling FLAC file: {e}")
            return None, None
        
        print("-------TTS transcript----------")
        print(script)
        print("-------------------------------")
        return audio_data_flac, script

    # Option 2: Load a random audio file from a public dataset - LibriSpeech for now
    else:
        subsets = ['train-clean-100', 'train-clean-360', 'train-other-500', 'dev-clean', 'dev-other', 'test-clean', 'test-other']
        selected_subset = random.choice(subsets)
        subset_path = os.path.join(base_path, 'LibriSpeech', selected_subset)
        
        # speaker_path = random.choice(glob.glob(os.path.join(subset_path, '*/*')))
        speaker_dirs = glob.glob(os.path.join(subset_path, '*/'))
        if not speaker_dirs:
            print(f"No speaker directories found in {subset_path}")
            return None, None
        speaker_dir = random.choice(speaker_dirs)
        
        chapter_dirs = glob.glob(os.path.join(speaker_dir, '*/'))
        if not chapter_dirs:
            print(f"No chapter directories found in {speaker_dir}")
            return None, None
        chapter_dir = random.choice(chapter_dirs)
        chapter_dir = chapter_dir.rstrip('/')
        path_parts = chapter_dir.split(os.sep)
        
        speaker_id = path_parts[-2] 
        chapter_id = path_parts[-1] 
        
        transcript_filename = f"{speaker_id}-{chapter_id}.trans.txt"
        transcript_file = os.path.join(chapter_dir, transcript_filename)

        if not os.path.exists(transcript_file):
            print(f"Transcript file not found: {transcript_file}")
            return None, None
        
        # Read the transcript file and choose a random line
        with open(transcript_file, 'r') as file:
            lines = file.read().strip().split('\n')
            line = random.choice(lines)
            audio_filename, transcript = line.split(' ', 1)

        # Construct the path to the corresponding audio file
        audio_filepath = os.path.join(chapter_dir, audio_filename + '.flac')
        if not os.path.exists(audio_filepath):
            print(f"Audio file not found: {audio_filepath}")
            return None, None

        # Read and return the audio data and its transcript
        with open(audio_filepath, 'rb') as file:
            audio_data = file.read()
        print("-------wave2vec transcript----------")
        print(transcript)
        print("----------------------------------")
        return audio_data, transcript

def google_tts(script, filename):
    try:
        tts = gTTS(script)
        tts.save(filename)
        return True
    except HTTPError as e:
        if e.response.status_code == 429:
            print("Hit rate limit for Google TTS")
            return False
        raise
    except gTTSError as e:
        if "429 (Too Many Requests)" in str(e):
            print("Hit rate limit for Google TTS")
            return False
        raise

def local_tts(script, filename):
    engine = pyttsx3.init()
    engine.save_to_file(script, filename)
    engine.runAndWait()

    # Wait for the file to be created
    timeout = 20  # Maximum number of seconds to wait
    start_time = time.time()
    while not os.path.exists(filename):
        time.sleep(1)
        if time.time() - start_time > timeout:
            print("Timeout waiting for TTS file to be created.")
            return False
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
