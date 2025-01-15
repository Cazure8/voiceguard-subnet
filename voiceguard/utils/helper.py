# The MIT License (MIT)
# Copyright © 2025 Yuma Rao
# Copyright © 2025 Cazure

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
import torch
import subprocess
import yt_dlp
import whisper
import re
import random
import requests

from io import BytesIO
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from uuid import uuid4
from dotenv import load_dotenv
from voiceguard.utils.misc import handle_filename_duplicates

load_dotenv()

proxy_url = os.getenv('PROXY_URL')
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
# Set pad token
tokenizer.pad_token = tokenizer.eos_token

def get_video_duration(url):
    ydl_opts = {
        'quiet': True,        # Suppresses most console output
        'no_warnings': True,  # Suppresses warnings
        'noplaylist': True,   # Ensures only a single video is processed
        'skip_download': True,  # No video download, just metadata
        'extract_flat': True,  # Faster metadata extraction without full processing
    }

    if proxy_url:
        ydl_opts['proxy'] = proxy_url 
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info_dict = ydl.extract_info(url, download=False)
            duration_seconds = info_dict.get('duration')
            if duration_seconds is None:
                raise ValueError("Failed to fetch video duration; the duration is None.")
            return duration_seconds
        except Exception as e:
            print(f"Error fetching video duration: {e}")
            return 0
    
def transcribe_with_whisper(audio_filepath):
    model = whisper.load_model("large")
    result = model.transcribe(audio_filepath)
    return result["text"]

def download_youtube_segment(youtube_url, segment, output_format='mp3', proxy=proxy_url):
    try:
        if not os.path.exists('download_youtube'):
            os.makedirs('download_youtube')

        file_uuid = uuid4()
        start_seconds, end_seconds = segment
        duration = end_seconds - start_seconds

        output_filename = f"{file_uuid}.{output_format}"
        output_filepath = os.path.join('download_youtube', output_filename)
        output_filepath = handle_filename_duplicates(output_filepath)

        command = [
            'yt-dlp',
            '-x',  # Extract audio
            '--audio-format', output_format,
            '--postprocessor-args',
            f"-ss {start_seconds} -t {duration} -ac 1 -ar 16000 -ab 128k",  # Segment extraction and conversion options
            '-o', output_filepath,
            '--quiet',
            youtube_url
        ]

        if proxy:
            command += ['--proxy', proxy]
       
        subprocess.run(command, check=True) 

        print(f"Segment audio downloaded and converted to {output_format}: {output_filepath}")
        return output_filepath
    
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def fetch_random_sentences(context="Voiceguard is the revolution.", num_sentences=3):
    # Tokenize input context
    input_ids = tokenizer.encode(context, return_tensors="pt")
    attention_mask = torch.ones_like(input_ids)  # Create attention mask

    # Generate text
    outputs = model.generate(
        input_ids,
        max_length=100,  # Ensure sufficient length for multiple sentences
        num_return_sequences=1,  # Generate a single long sequence
        do_sample=True,
        top_p=0.9,  # Adjust for slightly less randomness
        top_k=50,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode and clean text
    decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if context in decoded_text:
        decoded_text = decoded_text.replace(context, "")
        
    # Split the text into sentences using regular expressions
    split_sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', decoded_text)
    cleaned_sentences = [
        re.sub(r"[^a-zA-Z0-9.,'\" ]", "", s.strip())  # Remove unwanted characters
        for s in split_sentences
        if len(s.strip()) > 0
    ]

    # Concatenate the required number of sentences
    if len(cleaned_sentences) >= num_sentences:
        return " ".join(cleaned_sentences[:num_sentences])
    else:
        return " ".join(cleaned_sentences)  # Return all sentences if fewer are available
    
def get_random_audio_clip(directory="datasets/Common-Voice-Corpus-19.0_testsets"):
    # List all files in the directory
    files = os.listdir(directory)
    
    # Filter for .mp3 files
    audio_files = [f for f in files if f.endswith(".mp3")]
    
    # Check if there are any audio files
    if not audio_files:
        raise ValueError("No .mp3 files found in the directory.")
    
    # Select a random file
    random_file = random.choice(audio_files)
    
    # Return the full path of the random file
    return os.path.join(directory, random_file)

def get_random_asv_audio():
    url = "http://74.50.66.114:8000/asv-df"
    
    # Create "detection" directory if it doesn't exist
    detection_dir = "detection"
    os.makedirs(detection_dir, exist_ok=True)

    # Download the audio data
    response = requests.get(url)
    response.raise_for_status()

    # Parse the filename from the Content-Disposition header
    content_disposition = response.headers.get("Content-Disposition", "")
    match = re.search(r'filename="?([^"]+)"?', content_disposition)
    if match:
        filename = match.group(1)
    else:
        # Fallback name if none provided by server
        filename = "audio.mp3"

    # Determine if it's fake or real
    if "fake" in filename.lower():
        audio_type = "fake"
    else:
        audio_type = "real"

    # Save the file to the "detection" directory
    audio_file_stream = BytesIO(response.content)
    file_path = os.path.join(detection_dir, filename)
    with open(file_path, "wb") as f:
        f.write(audio_file_stream.getbuffer())

    # Return filename and audio_type
    return filename, audio_type