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
        output_filepath = download_youtube(audio_url)
        model, processor = load_model(output_filepath)
        waveform, sample_rate = read_audio(output_filepath)
        transcription = transcribe(model, processor, waveform, sample_rate)
        print("--------------------")
        print(transcription)
        print("--------------------")
        return transcription
        

def download_youtube(youtube_url, output_format='flac'):
    # Ensure the downloads directory exists
    download_dir = 'downloads'
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    yt = YouTube(youtube_url)
    audio_stream = yt.streams.filter(only_audio=True).first()
    
    # Generate the base filename for the download
    base_filename = yt.title.replace(" ", "_").replace("/", "_")
    output_filename = f"{base_filename}.{output_format}"
    output_filepath = os.path.join(download_dir, output_filename)
    
    # Handle potential filename duplicates
    counter = 1
    while os.path.exists(output_filepath):
        modified_base_filename = f"{base_filename}_{counter}"
        output_filename = f"{modified_base_filename}.{output_format}"
        output_filepath = os.path.join(download_dir, output_filename)
        counter += 1

    # Download the audio stream to a temporary file
    temp_filepath = audio_stream.download(output_path=download_dir, filename_prefix="temp_")
    
    # Convert the downloaded audio to the desired format using ffmpeg, if necessary
    if audio_stream.mime_type.split('/')[1] != output_format:
        converted_filepath = output_filepath
        subprocess.run(['ffmpeg', '-i', temp_filepath, '-vn', '-ar', '16000', '-ac', '1', '-ab', '192k', '-f', output_format, converted_filepath])
        os.remove(temp_filepath)  # Remove the temporary file after conversion
    else:
        # If the downloaded format is already .flac, just rename the file
        os.rename(temp_filepath, output_filepath)

    print(f"Audio downloaded and converted to .flac: {output_filepath}")
    return output_filepath

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
    
def load_model(file):
    # Predict the language
    prediction_result = recognition_model.classify_file(file)

    # Assuming the model returns a tuple where the last element contains language labels
    language_labels = prediction_result[-1]  # Adjust according to actual model output
    most_probable_language_label = language_labels[0]  # Assuming the first label is the most probable

    # Initialize model_id with a default model to ensure it's never empty
    default_model_id = "facebook/wav2vec2-base-960h"

    # Checking for Chinese variations
    if "Chinese" in most_probable_language_label:
        model_id = "ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt"
    elif "English" in most_probable_language_label:
        model_id = default_model_id
    else:
        model_id = default_model_id
        print(f"Unexpected language detected: {most_probable_language_label}. Using default model.")

    print(f"Detected language: {most_probable_language_label}, using model: {model_id}")

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



