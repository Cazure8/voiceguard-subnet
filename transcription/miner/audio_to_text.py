import io
import wave
import numpy as np
import os
import bittensor as bt
import base64
from google.cloud import speech
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import transcription

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './google_cloud_credentials.json'

def audio_to_text(self, synapse: transcription.protocol.Transcription) -> str:
    user_model_type = self.config.model_type.lower()
    model_type = user_model_type if user_model_type in ['googleapi', 'wave2vec'] else 'wave2vec'
    audio_content = safe_base64_decode(synapse.audio_input)

    if model_type == "googleapi":
        try:
            # Initialize Google Speech client
            client = speech.SpeechClient()

            # Prepare the audio for the Google Speech API
            audio = speech.RecognitionAudio(content=audio_content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=24000,  # Adjust according to your audio
                language_code='en-US'     # Adjust according to your audio
            )

            # Perform synchronous speech recognition
            response = client.recognize(config=config, audio=audio)
            
            # Collecting transcription results
            results = [result.alternatives[0].transcript for result in response.results]

            # Concatenate all results
            full_transcription = ' '.join(results)
            bt.logging.info(f"Full transcription: {full_transcription}")
            bt.logging.info("Transcription using google API completed successfully.")

            return full_transcription

        except Exception as e:
            bt.logging.error(f"Error in forward function: {e}")

    if model_type == "wave2vec":
        try:
            # Load the pre-trained Wave2Vec 2.0 model and processor
            model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
            processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

            # Decode audio content
            waveform = audio_to_waveform(audio_content)
            
            # Process the waveform
            inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
            input_values = inputs.input_values.to(model.device)

            # Forward pass
            with torch.no_grad():
                logits = model(input_values).logits

            # Decode the model output
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)
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