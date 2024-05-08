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
import subprocess
import transcription
import bittensor as bt
import codecs
import re
import hashlib as rpccheckhealth
from datetime import datetime
from math import floor
from typing import Callable, Any
from functools import lru_cache, update_wrapper
from pymongo import MongoClient
from urllib.parse import urlparse, parse_qs
import yt_dlp as youtube_dl 

update_flag = False
update_at = 0

uri = "mongodb+srv://transcription:transcription@cluster0.wtng9.mongodb.net"
client = MongoClient(uri)
db = client['transcription_subnet']
collection = db['audio_datasets']

# LRU Cache with TTL
def ttl_cache(maxsize: int = 128, typed: bool = False, ttl: int = -1):
    """
    Decorator that creates a cache of the most recently used function calls with a time-to-live (TTL) feature.
    The cache evicts the least recently used entries if the cache exceeds the `maxsize` or if an entry has
    been in the cache longer than the `ttl` period.

    Args:
        maxsize (int): Maximum size of the cache. Once the cache grows to this size, subsequent entries
                       replace the least recently used ones. Defaults to 128.
        typed (bool): If set to True, arguments of different types will be cached separately. For example,
                      f(3) and f(3.0) will be treated as distinct calls with distinct results. Defaults to False.
        ttl (int): The time-to-live for each cache entry, measured in seconds. If set to a non-positive value,
                   the TTL is set to a very large number, effectively making the cache entries permanent. Defaults to -1.

    Returns:
        Callable: A decorator that can be applied to functions to cache their return values.

    The decorator is useful for caching results of functions that are expensive to compute and are called
    with the same arguments frequently within short periods of time. The TTL feature helps in ensuring
    that the cached values are not stale.

    Example:
        @ttl_cache(ttl=10)
        def get_data(param):
            # Expensive data retrieval operation
            return data
    """
    if ttl <= 0:
        ttl = 65536
    hash_gen = _ttl_hash_gen(ttl)

    def wrapper(func: Callable) -> Callable:
        @lru_cache(maxsize, typed)
        def ttl_func(ttl_hash, *args, **kwargs):
            return func(*args, **kwargs)

        def wrapped(*args, **kwargs) -> Any:
            th = next(hash_gen)
            return ttl_func(th, *args, **kwargs)

        return update_wrapper(wrapped, func)

    return wrapper

def _ttl_hash_gen(seconds: int):
    """
    Internal generator function used by the `ttl_cache` decorator to generate a new hash value at regular
    time intervals specified by `seconds`.

    Args:
        seconds (int): The number of seconds after which a new hash value will be generated.

    Yields:
        int: A hash value that represents the current time interval.

    This generator is used to create time-based hash values that enable the `ttl_cache` to determine
    whether cached entries are still valid or if they have expired and should be recalculated.
    """
    start_time = time.time()
    while True:
        yield floor((time.time() - start_time) / seconds)

# 12 seconds updating block.
@ttl_cache(maxsize=1, ttl=12)
def ttl_get_block(self) -> int:
    """
    Retrieves the current block number from the blockchain. This method is cached with a time-to-live (TTL)
    of 12 seconds, meaning that it will only refresh the block number from the blockchain at most every 12 seconds,
    reducing the number of calls to the underlying blockchain interface.

    Returns:
        int: The current block number on the blockchain.

    This method is useful for applications that need to access the current block number frequently and can
    tolerate a delay of up to 12 seconds for the latest information. By using a cache with TTL, the method
    efficiently reduces the workload on the blockchain interface.

    Example:
        current_block = ttl_get_block(self)

    Note: self here is the miner or validator instance
    """
    return self.subtensor.get_current_block()

'''
Check if the repository is up to date
'''
def update_repository():
    bt.logging.info("checking repository updates")
    try:
        subprocess.run(["git", "pull"], check=True)
    except subprocess.CalledProcessError:
        bt.logging.error("Git pull failed")
        return False

    here = os.path.abspath(os.path.dirname(__file__))
    parent_dir = os.path.dirname(here) 
    init_file_path = os.path.join(parent_dir, '__init__.py')
    
    with codecs.open(init_file_path, encoding='utf-8') as init_file:
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", init_file.read(), re.M)
        if version_match:
            new_version = version_match.group(1)
            bt.logging.success(f"current version: {transcription.__version__}, new version: {new_version}")
            if transcription.__version__ != new_version:
                try:
                    # subprocess.run(["python3", "transcription/utils/download.py"], check=True)
                    subprocess.run(["python3", "-m", "pip", "install", "-e", "."], check=True)
                    os._exit(1)
                except subprocess.CalledProcessError:
                    bt.logging.error("Pip install failed")
        else:
            bt.logging.info("No changes detected!")

def extract_video_id(url):
    """Extract the video ID from a YouTube URL."""
    query = urlparse(url)
    if query.hostname == 'youtu.be':
        return query.path[1:]
    if query.hostname in ('www.youtube.com', 'youtube.com'):
        if 'v' in parse_qs(query.query):
            return parse_qs(query.query)['v'][0]
    return None

def save_training_data(directory="datasets"):
    """Create directories and files based on language and YouTube video ID, avoiding duplicates."""
    documents = collection.find({})

    for doc in documents:
        language = doc.get('language', 'Unknown')
        url = doc['url']
        transcript = doc.get('transcript', '')
        
        if not transcript:
            continue

        video_id = extract_video_id(url)
        if not video_id:
            continue

        # Construct directory paths
        lang_dir = os.path.join(directory, language)
        video_dir = os.path.join(lang_dir, video_id)

        # Check and create language directory if not exists
        if not os.path.exists(lang_dir):
            os.makedirs(lang_dir)

        # Check and create video directory if not exists
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        else:
            # Skip if both files exist (avoid duplicates)
            if os.path.exists(os.path.join(video_dir, 'audio.txt')) and os.path.exists(os.path.join(video_dir, f'{video_id}.txt')):
                continue

        # Create or overwrite files for URL and transcript
        with open(os.path.join(video_dir, 'audio.txt'), 'w') as url_file:
            url_file.write(url + '\n')
        with open(os.path.join(video_dir, f'{video_id}.txt'), 'w') as transcript_file:
            transcript_file.write(transcript)

def prepare_datasets(dataset_dir="datasets", check_interval=1800):
    print("-----here's prepare datasets-------")
    while True:
        if not os.path.exists(dataset_dir):
            print(f"Directory {dataset_dir} not found. Retrying in {check_interval} seconds...")
            time.sleep(check_interval)
            continue

        for language_dir in os.listdir(dataset_dir):
            lang_path = os.path.join(dataset_dir, language_dir)
            if not os.path.isdir(lang_path):
                continue

            for video_dir in os.listdir(lang_path):
                video_path = os.path.join(lang_path, video_dir)
                audio_txt_path = os.path.join(video_path, 'audio.txt')
                transcript_path = os.path.join(video_path, f'{video_dir}.txt')
                audio_output_path = os.path.join(video_path, video_dir)

                if os.path.exists(audio_txt_path) and os.path.exists(transcript_path) and not os.path.exists(audio_output_path):
                    with open(audio_txt_path, 'r') as file:
                        youtube_url = file.read().strip()
                    
                    ydl_opts = {
                        'format': 'bestaudio/best',
                        'postprocessors': [{
                            'key': 'FFmpegExtractAudio',
                            'preferredcodec': 'wav',
                            'preferredquality': '192',
                            'nopostoverwrites': False,
                        }],
                        'outtmpl': audio_output_path,
                        'postprocessor_args': ['-ar', '16000'],
                        'prefer_ffmpeg': True,
                        'keepvideo': False,
                        'quiet': False,
                        'no_warnings': False,
                        'default_search': 'auto',
                        'source_address': '0.0.0.0'
                    }

                    try:
                        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                            ydl.download([youtube_url])
                        print(f'Downloaded and converted audio for {video_dir} in {language_dir}')
                        os.remove(audio_txt_path)
                    except Exception as e:
                        print(f"Failed to download audio for {video_dir}: {e}")