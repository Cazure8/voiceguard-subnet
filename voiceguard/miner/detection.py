from voiceguard.protocol import VoiceGuardSynapse
import os
import librosa
import torch
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# Model directory
model_dir = "model_dir"

# Load the feature extractor and model
feature_extractor = AutoFeatureExtractor.from_pretrained(model_dir)
model = AutoModelForAudioClassification.from_pretrained(model_dir)

# Function to process and classify audio
def detect_deepfake(audio_file):
    """
    Detects if the given audio file is a deepfake or real.
    
    Args:
        audio_file (str): Path to the audio file.

    Returns:
        dict: Dictionary with class probabilities and the predicted label.
    """
    # Load audio file and resample to 16kHz
    audio, sampling_rate = librosa.load(audio_file, sr=16000)
    
    # Prepare input features
    inputs = feature_extractor(audio, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    
    # Run inference
    with torch.no_grad():
        logits = model(**inputs).logits
    
    # Convert logits to probabilities
    probabilities = F.softmax(logits, dim=-1)
    
    # Get the predicted label
    predicted_label = torch.argmax(probabilities, dim=-1).item()
    label_name = model.config.id2label[predicted_label]
    
    # Prepare result
    result = {
        "predicted_label": label_name,
        "class_probabilities": probabilities.squeeze().tolist(),
    }
    return result

if __name__ == "__main__":
    # Path to the audio file to classify
    audio_file_path = 'one_clip.wav'
    
    if not os.path.isfile(audio_file_path):
        print(f"Error: The file {audio_file_path} does not exist.")
    else:
        result = detect_deepfake(audio_file_path)
        print("\n=== Detection Results ===")
        print(f"Predicted Label: {result['predicted_label']}")
        print("Class Probabilities:")
        for i, prob in enumerate(result["class_probabilities"]):
            print(f"  {model.config.id2label[i]}: {prob:.4f}")
