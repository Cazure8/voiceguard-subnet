import re
import os
import subprocess
import bittensor as bt
import torch
import torchaudio
from transcription.protocol import Transcription
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datetime import datetime
from transcription.protocol import Transcription
from pytube import YouTube
from speechbrain.pretrained import SpeakerRecognition

recognition_model = SpeakerRecognition.from_hparams(source="speechbrain/lang-id-commonlanguage_ecapa", savedir="tmpdir")

def url_to_text(self, synapse: Transcription) -> str:
    audio_url = synapse.audio_input
    if is_twitter_space(audio_url):
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        downloaded = download_twitter_space(audio_url, timestamp)
        bt.logging.info("Downloaded successfully!" if downloaded else "Download failed!")
        transcription = ""
        
        return transcription
    
    elif is_youtube(audio_url):
        filename = download_youtube(audio_url)
        model, processor = load_model(filename)
        output_file = os.path.join("downloads", filename)
        waveform, sample_rate = read_audio(output_file)
        transcription = transcribe(model, processor, waveform, sample_rate)

        return transcription
        

def download_youtube(youtube_url, output_format='flac'):
    yt = YouTube(youtube_url)
    stream = yt.streams.filter(only_audio=True).first()

    if not os.path.exists('downloads'):
        os.makedirs('downloads')

    base = yt.title
    filename = f"{base}.{output_format}"
    output_file = os.path.join("downloads", filename)
    
    # Check for duplicate filename and modify if necessary
    counter = 1
    while os.path.exists(output_file):
        new_filename = f"{base}_{counter}.{output_format}"
        output_file = os.path.join("downloads", new_filename)
        counter += 1

    # Download the file
    stream.download(output_path='downloads', filename=os.path.basename(output_file))

    return os.path.basename(output_file)

def is_twitter_space(url):
    pattern = r'https://twitter\.com/i/spaces/\S+'
    return re.match(pattern, url) is not None

def is_youtube(url):
    # Expanded pattern to cover more YouTube URL variations including channels, user pages, and additional query parameters
    pattern = r'(https?://)?(www\.)?(youtube\.com/watch\?v=|youtube\.com/playlist\?list=|youtube\.com/channel/|youtube\.com/user/|youtu\.be/)[\w-]+(&[\w-]+=[\w-]+)*'
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
    
def load_model(filename):
    audio_file = os.path.join("downloads", filename)

    # Predict the language
    prediction_result = recognition_model.classify_file(audio_file)

    # Extract the logits tensor for completeness, in case it's needed for further analysis
    logits_tensor = prediction_result[0]

    _, _, _, language_labels = prediction_result
    most_probable_language = language_labels[0]  # Assuming the model returns a list with the label

    # Initialize model_id with a default model to ensure it's never empty
    model_id = "facebook/wav2vec2-base-960h"  # Default model for English or as a fallback

    if most_probable_language in ["Chinese_Taiwan", "Chinese_Hongkong", "Chinese_Simplified", "Chinese_Traditional"]:
        model_id = "ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt"
    elif most_probable_language == "English":
        pass
    else:
        print(f"Unexpected language detected: {most_probable_language}. Using default model.")

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



