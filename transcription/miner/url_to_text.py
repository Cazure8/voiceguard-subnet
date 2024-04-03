import re
import os
import subprocess
import bittensor as bt
import torch
import torchaudio
from transcription.protocol import Transcription
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datetime import datetime
from speechbrain.pretrained import SpeakerRecognition
from uuid import uuid4
import whisper

recognition_model = SpeakerRecognition.from_hparams(source="speechbrain/lang-id-commonlanguage_ecapa", savedir="tmpdir")

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
    formatted_transcription = f"{segment_start}$$__{transcription}"
    return formatted_transcription

def download_youtube_segment(youtube_url, segment, output_format='mp3'):
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
            youtube_url
        ]

        subprocess.run(command, check=True) 

        print(f"Segment audio downloaded and converted to {output_format}: {output_filepath}")
        return output_filepath
    
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def convert_seconds_to_time(seconds):
    """Convert seconds to HH:MM:SS format."""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

def handle_filename_duplicates(filepath):
    """Ensure the filepath is unique to avoid overwriting existing files."""
    base, extension = os.path.splitext(filepath)
    counter = 1
    while os.path.exists(filepath):
        filepath = f"{base}_{counter}{extension}"
        counter += 1
    return filepath

def transcribe_with_whisper(audio_filepath):
    model = whisper.load_model("medium")
    result = model.transcribe(audio_filepath)
    return result["text"]

def is_twitter_space(url):
    pattern = r'https://twitter\.com/i/spaces/\S+'
    return re.match(pattern, url) is not None

def is_youtube(url):
    pattern = r'(https?://)?(www\.|m\.)?(youtube\.com/watch\?v=|youtube\.com/playlist\?list=|youtube\.com/channel/|youtube\.com/user/|youtu\.be/)[\w-]+(\?[\w=&-]*)?'
    return re.match(pattern, url) is not None

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
    
def load_model(file):
    default_model_id = "facebook/wav2vec2-base-960h"
    model_id = default_model_id  
    most_probable_language_label = "Unknown" 

    try:
        # Predict the language
        prediction_result = recognition_model.classify_file(file)

        # Assuming the model returns a tuple where the last element contains language labels
        language_labels = prediction_result[-1]  # Adjust according to actual model output
        most_probable_language_label = language_labels[0]  # Assuming the first label is the most probable
        
        # Checking for specific language variations
        if "Chinese" in most_probable_language_label:
            model_id = "ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt"
        elif "English" in most_probable_language_label:
            model_id = default_model_id
        else:
            print(f"Unexpected language detected: {most_probable_language_label}. Using default model.")
            
        print(f"Detected language: {most_probable_language_label}, using model: {model_id}")

    except Exception as e:
        print(f"An error occurred during language prediction: {e}. Falling back to default English model.")

    # Load the model and processor using the determined model_id
    model = Wav2Vec2ForCTC.from_pretrained(model_id)
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    
    return model, processor

def read_audio(file_path, target_sample_rate=16000):
    # Load the audio file
    waveform, sample_rate = torchaudio.load(file_path)

    # Print shape of the waveform for debugging
    print("Original waveform shape:", waveform.shape)

    # Convert stereo to mono by averaging channels if necessary
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)

    # Resample the waveform to the target sample rate if necessary
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)

    # Ensure the waveform is 2D [1, sequence_length]
    waveform = waveform.unsqueeze(0) if waveform.dim() == 1 else waveform

    # Print shape of the waveform after processing
    print("Processed waveform shape:", waveform.shape)

    return waveform, target_sample_rate

def transcribe(model, processor, waveform, sample_rate):
    # Preprocess the waveform to match the model input
    input_values = processor(waveform, sampling_rate=sample_rate, return_tensors="pt").input_values

    # Remove the extra dimension from input_values
    input_values = input_values.squeeze(1)

    # Print shape of the input_values after modification
    print("Modified input values shape for model:", input_values.shape)

    # Perform inference
    with torch.no_grad():
        logits = model(input_values).logits

    # Decode the model output
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    return transcription[0]

def check_urls(file_path):
    valid_urls = []
    with open(file_path, 'r') as file:
        urls = file.readlines()
    
    for url in urls:
        try:
            yt = YouTube(url.strip())
            print(f"URL is valid: {url}")
            valid_urls.append(url.strip())
        except Exception as e:
            print(f"URL is not valid or has restrictions: {url}. Reason: {e}")
    
    # Optionally, save the valid URLs to a new file
    with open('valid_youtube_urls.txt', 'w') as valid_urls_file:
        for url in valid_urls:
            valid_urls_file.write(url + '\n')
    
    print(f"Total valid URLs: {len(valid_urls)}")
    return valid_urls