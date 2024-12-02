# The MIT License (MIT)
# Copyright © 2024 Cazure

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
from voiceguard.protocol import VoiceGuardSynapse
from voiceguard.validator.stt_reward import get_stt_rewards
from voiceguard.validator.clone_reward import get_clone_rewards
from voiceguard.validator.detection_reward import get_detection_rewards
from voiceguard.utils.uids import get_random_uids
import random
import torchaudio.transforms as T
import torch
import soundfile as sf
from voiceguard.utils.stt_helper import download_youtube_segment, transcribe_with_whisper, get_video_duration
from voiceguard.utils.misc import select_random_url

async def forward(self):
    """
    The forward function is called by the validator every time step.

    It is responsible for querying the network and scoring the responses.

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

    """
    miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)
    # miner_uids = [3]

    try:
        # equally randomly send requests for tts, cloning, deepfake
        if random.random() < 0.333: # tts requests
            random_url = select_random_url()
            duration = get_video_duration(random_url)
            validator_segment = generate_validator_segment(duration)
            output_filepath = download_youtube_segment(random_url, validator_segment)

            if not os.path.exists(output_filepath):
                print("Output file does not exist. Returning empty transcription.")
                transcription = ""
            else:
                transcription = transcribe_with_whisper(output_filepath)

            print("------validator transcript--------")
            print(transcription)
            print("----------------------------------")
            responses = self.dendrite.query(
                # Send the query to selected miner axons in the network.
                axons=[self.metagraph.axons[uid] for uid in miner_uids],
                synapse = VoiceGuardSynapse(synapse_type="stt", stt_link=random_url, stt_segment=validator_segment),
                deserialize=False,
                timeout=50
            )
            
            rewards = get_stt_rewards(self, query=transcription, responses=responses, time_limit=50)
            self.update_scores(rewards, miner_uids, score_type="stt")
            
        elif random.random() < 0.667: # clone request
            # read and get clone_audio segment and text to clone from DB
            clone_text = "This is sample text to clone from DB like public one-WANBD"
            
            clone_audio_path = "/commonvoice/clone_clip.mp3"  # Path to the .mp3 file
            
            with open(clone_audio_path, "rb") as audio_file:
                clone_clip = audio_file.read() 
                
            responses = self.dendrite.query(
                # Send the query to selected miner axons in the network.
                axons=[self.metagraph.axons[uid] for uid in miner_uids],
                synapse = VoiceGuardSynapse(synapse_type="clone", clone_clip_path=clone_clip, clone_text=clone_text),
                deserialize=False,
                timeout=50
            )
            
            rewards = get_clone_rewards(self, clone_clip_path=clone_audio_path, responses=responses, clone_text=clone_text, time_limit=50)
            self.update_scores(rewards, miner_uids, score_type="clone")
            
        elif random.random() < 1:
            # get real audio or fake audio from DB
            random_audio_path = "/detection/random.mp3"
            random_audio_type = "real"
            
            with open(random_audio_path, "rb") as audio_file:
                random_audio = audio_file.read()
            
            responses = self.dendrite.query(
                # Send the query to selected miner axons in the network.
                axons=[self.metagraph.axons[uid] for uid in miner_uids],
                synapse = VoiceGuardSynapse(synapse_type="detection", detection_audio=random_audio),
                deserialize=False,
                timeout=50
            )
            
            rewards = get_detection_rewards(self, audio_type = random_audio_type, responses=responses, time_limit=50)
            self.update_scores(rewards, miner_uids, score_type="detection")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        rewards = torch.zeros(len(miner_uids))
        
    bt.logging.info(f"Scored responses: {rewards}")
    
    

def generate_validator_segment(duration):
    if duration <= 100:
        return [0, duration]
    else:
        start = random.randint(0, duration - 100)
        return [start, start + 100]