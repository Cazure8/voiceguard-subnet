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
import random
import io
import bittensor as bt
import base64
import glob
import torchaudio
from pydub import AudioSegment

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
    # if random.choice([True, False]):
    if True:
        script = generate_random_text(num_sentences=5, sentence_length=5)
        tts = gTTS(script)
        mp3_file_name = "temp_{}.mp3".format(random.randint(0, 10000))
        tts.save(mp3_file_name)
        
        audio_file = "output_audio_{}.wav".format(random.randint(0, 10000))
        sound = AudioSegment.from_mp3(mp3_file_name)
        sound.export(audio_file, format="wav")
        os.remove(mp3_file_name)
        
        with open(audio_file, 'rb') as file:
            audio_data = file.read()
        os.remove(audio_file)  # Clean up the generated file
        return audio_data, script

    # Option 2: Load a random audio file from a public dataset - LibriSpeech for now
    else:
        subsets = ['train-clean-100', 'train-clean-360', 'train-other-500', 'dev-clean', 'dev-other', 'test-clean', 'test-other']
        selected_subset = random.choice(subsets)
        subset_path = os.path.join(base_path, selected_subset)

        speaker_path = random.choice(glob.glob(os.path.join(subset_path, '*/*')))
        chapter_path = random.choice(glob.glob(os.path.join(speaker_path, '*')))
        transcript_path = os.path.join(chapter_path, f"{os.path.basename(chapter_path)}.trans.txt")
        with open(transcript_path, 'r') as file:
            lines = file.readlines()
            selected_line = random.choice(lines).strip()
            audio_file, transcript = selected_line.split(' ', 1)
            audio_file_path = os.path.join(chapter_path, f"{audio_file}.flac")
        print("--------transcript---------")
        print(transcript)
        print("---------------------------")
        with open(audio_file_path, 'rb') as file:
            audio_data = file.read()

        binary_audio_data = waveform_to_binary(audio_data, 16000)
        
        return binary_audio_data, transcript

def waveform_to_binary(waveform, sample_rate):
    """
    Converts a waveform tensor back to binary audio data.
    """
    binary_stream = io.BytesIO()
    torchaudio.save(binary_stream, waveform, sample_rate)
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
