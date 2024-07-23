import re
import os
import subprocess
import bittensor as bt
from voiceguard.protocol import Transcription
from datetime import datetime
from voiceguard.utils.misc import is_twitter_space, is_youtube
from voiceguard.utils.transcribe_manage import download_youtube_segment, transcribe_with_whisper

def url_to_text(self, synapse: Transcription) -> str:
    audio_url = synapse.audio_input
    segment = synapse.segment

    if is_twitter_space(audio_url):
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        downloaded = download_twitter_space(audio_url, timestamp)
        bt.logging.info("Downloaded successfully!" if downloaded else "Download failed!")
        transcription = ""
        
        return transcription
    
    elif is_youtube(audio_url):
        try:
            output_filepath = download_youtube_segment(audio_url, segment)
            
            if not os.path.exists(output_filepath):
                print("Output file does not exist. Returning empty transcription.")
                start, _ = segment
                return format_transcription(start, "")
            
            transcription = transcribe_with_whisper(output_filepath)

            print("---miner transcript--")
            print(transcription)
            print("---------------------")

            start, _ = segment
            return format_transcription(start, transcription)
        
        except Exception as e:
            print(f"Failed during model loading or transcription: {e}")
            return ""

def format_transcription(segment_start, transcription):
    formatted_transcription = f"{segment_start}$$_{transcription}"
    return formatted_transcription

# def download_youtube_segment(youtube_url, segment, output_format='mp3', proxy=proxy_url):
#     try:
#         if not os.path.exists('downloads'):
#             os.makedirs('downloads')

#         file_uuid = uuid4()
#         start_seconds, end_seconds = segment
#         duration = end_seconds - start_seconds

#         output_filename = f"{file_uuid}.{output_format}"
#         output_filepath = os.path.join('downloads', output_filename)
#         output_filepath = handle_filename_duplicates(output_filepath)

#         command = [
#             'yt-dlp',
#             '-x',  # Extract audio
#             '--audio-format', output_format,
#             '--postprocessor-args',
#             f"-ss {start_seconds} -t {duration} -ac 1 -ar 16000 -ab 128k",  # Segment extraction and conversion options
#             '-o', output_filepath,
#             youtube_url
#         ]
#         print(f"Proxy value inside function: {proxy}")  # Debug print
#         if proxy:
#             command += ['--proxy', proxy]
       
#         subprocess.run(command, check=True) 

#         print(f"Segment audio downloaded and converted to {output_format}: {output_filepath}")
#         return output_filepath
    
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")

# def handle_filename_duplicates(filepath):
#     """Ensure the filepath is unique to avoid overwriting existing files."""
#     base, extension = os.path.splitext(filepath)
#     counter = 1
#     while os.path.exists(filepath):
#         filepath = f"{base}_{counter}{extension}"
#         counter += 1
#     return filepath

# def transcribe_with_whisper(audio_filepath):
#     model = whisper.load_model("large")
#     result = model.transcribe(audio_filepath)
#     return result["text"]

def download_twitter_space(url, output):
    try:
        # Set the path for FFMPEG if it's not in the default PATH
        if "FFMPEG_BIN_PATH" in os.environ:
            os.environ["PATH"] += os.pathsep + os.getenv("FFMPEG_BIN_PATH")

        # Ensure downloads directory exists
        os.makedirs("downloads", exist_ok=True)
        
        # Construct the command to download the Twitter Space
        command = ["twspace_dl", "-i", url, "-o", f"downloads/{output}"]
        print(command)
        # Start the download process
        download_process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Wait for the process to complete and capture output
        stdout, stderr = download_process.communicate()

        if download_process.returncode != 0:
            print(f"Error downloading Twitter Space: {stderr.decode().strip()}")
            return False

        return True
    except Exception as e:
        print(f"An exception occurred while downloading Twitter space: {e}")
        return False