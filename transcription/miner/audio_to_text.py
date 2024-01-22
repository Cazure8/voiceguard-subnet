import io
import wave
import numpy as np
import os
import bittensor as bt
import base64
from google.cloud import speech
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import torchaudio
import transcription

global_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
global_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

def audio_to_text(self, synapse: transcription.protocol.Transcription) -> str:
    audio_content = safe_base64_decode(synapse.audio_input)

    try:
        # Load the audio and convert it to a waveform
        waveform, sample_rate = torchaudio.load(io.BytesIO(audio_content))

        # If waveform is stereo (2 channels), convert it to mono
        if waveform.shape[0] == 2:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Ensure waveform is 2D [batch_size, sequence_length]
        if waveform.dim() != 2:
            raise ValueError(f"Unexpected waveform dimension: {waveform.dim()}")

        # Resample if necessary
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

        # Process the waveform
        inputs = global_processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt", padding=True)
        input_values = inputs.input_values

        if input_values.dim() != 2:
            raise ValueError(f"Unexpected input values dimension after processing: {input_values.dim()}")

        input_values = input_values.to(global_model.device)

        # Forward pass
        with torch.no_grad():
            logits = global_model(input_values).logits

        # Decode the model output
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = global_processor.batch_decode(predicted_ids)
        print("-------miner transcript------------")
        print(transcription[0])
        print("-----------------------------------")
        return transcription[0]
    
    except Exception as e:
        bt.logging.error(f"Error in Wave2Vec transcription: {e}")
        return "Error during transcription"
    
def safe_base64_decode(data):
        """Safely decode a base64 string, ensuring correct padding."""
        padding = len(data) % 4
        if padding != 0:
            data += '=' * (4 - padding)
        return base64.b64decode(data)

def audio_to_waveform(audio_content):
    """
    Converts audio content in WAV format to a waveform.
    """
    # Read WAV from bytes
    with io.BytesIO(audio_content) as audio_file:
        with wave.open(audio_file, 'rb') as wav:
            n_channels, sample_width, framerate, n_frames, _, _ = wav.getparams()
            frames = wav.readframes(n_frames)

    # Convert byte data to numpy array
    waveform = np.frombuffer(frames, dtype=np.int16)
    
    # Normalize waveform to be in [-1, 1]
    waveform = waveform / np.iinfo(np.int16).max

    return torch.tensor(waveform).unsqueeze(0)