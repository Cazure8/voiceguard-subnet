import base64
import io
import librosa
import torch
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from voiceguard.protocol import VoiceGuardSynapse


# Load the model 
model_dir = "./pretrained/model_dir"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_dir)
model = AutoModelForAudioClassification.from_pretrained(model_dir)

def deepfake_detection(self, synapse: VoiceGuardSynapse) -> bytes:
    """
    Detects if the given audio is a deepfake using the VoiceGuardSynapse protocol.
    
    Args:
        synapse: VoiceGuardSynapse object containing the audio data
        
    Returns:
        bytes: Detection probability as bytes
    """
    try:
        # Get audio data from synapse
        detection_audio_bytes = base64.b64decode(synapse.detection_audio)
        
        
        # Convert bytes to audio array using librosa
        audio, sampling_rate = librosa.load(io.BytesIO(detection_audio_bytes), sr=16000)
        
        # Prepare input features
        inputs = feature_extractor(audio, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        
        # Run inference
        with torch.no_grad():
            logits = model(**inputs).logits
        
        # Get probability of deepfake
        probabilities = F.softmax(logits, dim=-1)
        # Audio returning probability > 0.5 is likely to be deepfake else likely to be real
        detection_prediction = probabilities[0][1].item() 
        
        # Convert to bytes and return
        return str(detection_prediction).encode()
        
    except Exception as e:
        # Handle errors by returning a default value
        print(f"Error in deepfake detection: {str(e)}")
        return str(0.0).encode()