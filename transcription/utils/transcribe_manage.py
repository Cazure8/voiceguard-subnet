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
import subprocess
from pytube import YouTube
from uuid import uuid4
import whisper
from dotenv import load_dotenv
from transcription.utils.misc import handle_filename_duplicates

load_dotenv()

proxy_url = os.getenv('PROXY_URL')

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
    
def transcribe_with_whisper(audio_filepath):
    model = whisper.load_model("large")
    result = model.transcribe(audio_filepath)
    return result["text"]

def download_youtube_segment(youtube_url, segment, output_format='mp3', proxy=proxy_url):
    try:
        if not os.path.exists('downloads'):
            os.makedirs('downloads')

        file_uuid = uuid4()
        start_seconds, end_seconds = segment
        duration = end_seconds - start_seconds

        output_filename = f"{file_uuid}.{output_format}"
        output_filepath = os.path.join('downloads', output_filename)
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