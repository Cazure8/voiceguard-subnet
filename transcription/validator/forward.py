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
from pytube import YouTube
from requests.exceptions import HTTPError
from transcription.protocol import Transcription
from transcription.validator.reward import get_rewards
from transcription.utils.uids import get_random_uids
from gtts import gTTS, gTTSError
import random
import torchaudio.transforms as T
import torch
import soundfile as sf

from transcription.miner.url_to_text import download_youtube_segment, load_model, read_audio, transcribe

async def forward(self):
    """
    The forward function is called by the validator every time step.

    It is responsible for querying the network and scoring the responses.

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

    """
    miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)
    if random.random() < 0.6:
        audio_sample, ground_truth_transcription = generate_or_load_audio_sample()
        audio_sample_base64 = encode_audio_to_base64(audio_sample)
    
        # The dendrite client queries the network.
        responses = self.dendrite.query(
            # Send the query to selected miner axons in the network.
            axons=[self.metagraph.axons[uid] for uid in miner_uids],
            synapse=Transcription(audio_input=audio_sample_base64),
            deserialize=False,
        )
        rewards = get_rewards(self, query=ground_truth_transcription, responses=responses, type="not_url", time_limit=12)

    else:
        random_url = select_random_url('youtube_urls.txt')
        duration = get_video_duration(random_url)
        validator_segment = generate_validator_segment(duration)
        synapse_segment = generate_synapse_segment(duration, validator_segment[0])

        #TODO: refactoring functions required
        output_filepath = download_youtube_segment(random_url, validator_segment)
        model, processor = load_model(output_filepath)
        waveform, sample_rate = read_audio(output_filepath)
        transcription = transcribe(model, processor, waveform, sample_rate)

        responses = self.dendrite.query(
            # Send the query to selected miner axons in the network.
            axons=[self.metagraph.axons[uid] for uid in miner_uids],
            synapse = Transcription(input_type="url", audio_input=random_url, segment=validator_segment),
            deserialize=False,
            timeout=50
        )

        rewards = get_rewards(self, query=transcription, responses=responses, type="url", time_limit=60)

    bt.logging.info(f"Scored responses: {rewards}")
    # Update the scores based on the rewards. You may want to define your own update_scores function for custom behavior.
    self.update_scores(rewards, miner_uids)
    
def generate_or_load_audio_sample(base_path='librispeech_dataset'):
    # Attempt to generate audio using TTS
    script = generate_random_text(num_sentences=25, sentence_length=10)
    mp3_file_name = f"temp_{random.randint(0, 10000)}.mp3"

    if random.random() < 0.2 and (google_tts(script, mp3_file_name) or local_tts(script, mp3_file_name)):
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
    
def iterate_audio_files(subset_path):
    """
    Iterates over audio files and applies a random augmentation.
    """
    for speaker_dir in glob.glob(os.path.join(subset_path, '*/')):
        for chapter_dir in glob.glob(os.path.join(speaker_dir, '*/')):
            transcript_files = glob.glob(os.path.join(chapter_dir, "*.trans.txt"))
            if transcript_files:
                transcript_file = transcript_files[0]
                with open(transcript_file, 'r') as file:
                    for line in file.readlines():
                        audio_filename, transcript = line.strip().split(' ', 1)
                        audio_filepath = os.path.join(chapter_dir, audio_filename + '.flac')
                        if os.path.exists(audio_filepath):
                            audio_data = random_audio_augmentation(audio_filepath)
                            # Return or process the augmented waveform and its transcript
                            return audio_data, transcript
    return None, None

def load_from_librispeech(base_path):
    subsets = ['train-clean-100', 'train-clean-360', 'train-other-500', 'dev-clean', 'dev-other', 'test-clean', 'test-other']
    for subset in subsets:
        subset_path = os.path.join(base_path, 'LibriSpeech', subset)
        audio_data, transcript = iterate_audio_files(subset_path)  # This now expects audio data, not a path
        if audio_data:
            return audio_data, transcript
    print("No valid audio files found in the dataset.")
    return None, None

def random_audio_augmentation(audio_filepath, sample_rate=16000):
    """
    Applies a random augmentation (pitch, speed, noise) to the audio data.
    """
    waveform, original_sample_rate = torchaudio.load(audio_filepath)

    # Resample waveform to target sample rate if necessary
    if original_sample_rate != sample_rate:
        resampler = T.Resample(orig_freq=original_sample_rate, new_freq=sample_rate)
        waveform = resampler(waveform)

    # Randomly choose an augmentation
    augmentation_choice = random.choice(["original", "pitch", "speed", "noise"])

    if augmentation_choice == "pitch":
        # Apply pitch shift
        n_steps = random.randint(-2, 2)  # Pitch shift by up to 2 semitones
        pitch_shift = n_steps * 100  # Convert semitones to cents
        effects = [['pitch', str(pitch_shift)]]
        waveform, _ = torchaudio.sox_effects.apply_effects_tensor(waveform, sample_rate, effects)

    elif augmentation_choice == "speed":
        # Apply speed change
        speed_factor = random.uniform(0.9, 1.1)  # Speed up or slow down by up to 10%
        effects = [['speed', str(speed_factor)]]
        waveform, _ = torchaudio.sox_effects.apply_effects_tensor(waveform, sample_rate, effects)

    elif augmentation_choice == "noise":
        # Add noise
        noise_intensity = random.uniform(0.001, 0.005)  # Adjust intensity of noise
        noise = torch.randn_like(waveform) * noise_intensity
        waveform += noise
    
    return waveform_to_binary(waveform)

def waveform_to_binary(waveform, sample_rate=16000, format='FLAC'):
    """
    Converts a waveform to binary audio data in memory as FLAC.
    """
    # Create an in-memory binary stream
    audio_binary = io.BytesIO()
    
    # Convert the tensor waveform to numpy array
    waveform_np = waveform.numpy()
    
    # Save waveform to the binary stream as FLAC
    sf.write(audio_binary, waveform_np.T, sample_rate, format=format)
    
    # Seek to the start so it can be read from
    audio_binary.seek(0)
    
    # Read the binary audio data from the stream
    audio_data = audio_binary.read()
    
    return audio_data

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
        
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            return True
        else:
            print("TTS API returned OK, but the audio file is empty or not created.")
            return False
    except HTTPError as e:
        if e.response.status_code == 429:
            print("Hit rate limit for Google TTS")
        elif e.response.status_code == 500:
            print("Internal Server Error from TTS API")
        elif e.response.status_code == 503:
            print("Service Unavailable. TTS API might be down or undergoing maintenance")
        elif e.response.status_code == 401:
            print("Unauthorized. Check your API key or authentication method")
        elif e.response.status_code == 403:
            print("Forbidden. You might not have permission to use this service")
        else:
            print(f"HTTP error occurred: {e.response.status_code} {e.response.reason}")
        return False
    except gTTSError as e:
        if "429 (Too Many Requests)" in str(e):
            print("Hit rate limit for Google TTS")
        elif "500 (Internal Server Error)" in str(e):
            print("Internal Server Error from TTS API. Probably cause: Upstream API error")
        elif "Failed to connect" in str(e):
            print("Connection error in Google TTS")
        else:
            print("Unknown gTTSError")
        return False
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

def generate_random_sentence(words, length=10):
    return ' '.join(random.choices(words, k=length))

def generate_random_text(num_sentences=5, sentence_length=5):
    common_words = ['THE', 'BE', 'TO', 'OF', 'AND', 'A', 'IN', 'THAT', 'HAVE', 'I',
                    'IT', 'FOR', 'NOT', 'ON', 'WITH', 'HE', 'AS', 'YOU', 'DO', 'AT',
                    'THIS', 'BUT', 'HIS', 'BY', 'FROM', 'THEY', 'WE', 'SAY', 'HER',
                    'SHE', 'OR', 'AN', 'WILL', 'MY', 'ONE', 'ALL', 'WOULD', 'THERE',
                    'THEIR', 'WHAT', 'SO', 'UP', 'OUT', 'IF', 'ABOUT', 'WHO', 'GET',
                    'WHICH', 'GO', 'ME', 'WHEN', 'MAKE', 'CAN', 'LIKE', 'TIME', 'NO',
                    'JUST', 'HIM', 'KNOW', 'TAKE', 'PERSON', 'INTO', 'YEAR', 'YOUR',
                    'GOOD', 'SOME', 'COULD', 'THEM', 'SEE', 'OTHER', 'THAN', 'THEN',
                    'NOW', 'LOOK', 'ONLY', 'COME', 'ITS', 'OVER', 'THINK', 'ALSO',
                    'BACK', 'AFTER', 'USE', 'TWO', 'HOW', 'OUR', 'WORK', 'FIRST',
                    'WELL', 'WAY', 'EVEN', 'NEW', 'WANT', 'BECAUSE', 'ANY', 'THESE',
                    'GIVE', 'DAY', 'MOST', 'US']

    return ' '.join([generate_random_sentence(common_words, sentence_length) for _ in range(num_sentences)])

def encode_audio_to_base64(audio_data):
    # Encode binary audio data to Base64 string
    return base64.b64encode(audio_data).decode('utf-8')

def get_video_duration(url):
    try:
        # Create a YouTube object with the URL
        yt = YouTube(url)
        
        # Fetch the duration of the video in seconds
        duration_seconds = yt.length
        return duration_seconds
    except Exception as e:
        print(f"Error fetching video duration: {e}")
        return 0
    
def select_random_url(filename):
    with open(filename, 'r') as file:
        urls = file.readlines()
    return random.choice(urls).strip()

def generate_validator_segment(duration):
    if duration <= 100:
        return [0, duration]
    else:
        start = random.randint(0, duration - 100)
        return [start, start + 100]
    
def generate_synapse_segment(duration, validator_start):
    start = max(0, validator_start - 50)
    end = min(duration, validator_start + 150)
    return [start, end]