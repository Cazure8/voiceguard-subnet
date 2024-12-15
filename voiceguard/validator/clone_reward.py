import os
import time
import base64
import torchaudio
import numpy as np
import librosa
from pydub import AudioSegment
from typing import List
from scipy.spatial.distance import cosine, euclidean
from speechbrain.pretrained import SpeakerRecognition
from voiceguard.utils.helper import transcribe_with_whisper
from voiceguard.validator.mos_net.mosnet import MOSNet
from voiceguard.validator.stt_reward import overall_correctness_score

# Global variables for preloaded models
VERIFICATION_MODEL = None
MOSNET_MODEL = None

# Directory to save cloned audio files
CLONED_AUDIO_DIR = "miner_cloned_voices"
os.makedirs(CLONED_AUDIO_DIR, exist_ok=True)  # Ensure the directory exists

# ---------------------------
# Model Loading Functions
# ---------------------------

def get_verification_model():
    """Load the SpeakerRecognition model once and reuse it."""
    global VERIFICATION_MODEL
    if VERIFICATION_MODEL is None:
        VERIFICATION_MODEL = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    return VERIFICATION_MODEL

def get_mosnet_model():
    """Load the MOSNet model once and reuse it."""
    global MOSNET_MODEL
    if MOSNET_MODEL is None:
        model_path = "./pretrained/cnn_blstm.h5"
        MOSNET_MODEL = MOSNet(pretrained_model_path=model_path)
    return MOSNET_MODEL


# ---------------------------
# Audio Processing Utilities
# ---------------------------

def convert_to_wav(input_path, output_path=None):
    """Convert an MP3 or other audio format to WAV."""
    audio = AudioSegment.from_file(input_path)
    if not output_path:
        output_path = input_path.replace(".mp3", ".wav")
    audio.export(output_path, format="wav")
    return output_path

def get_audio_embeddings(audio_path: str, model: SpeakerRecognition) -> np.ndarray:
    """Generate speaker embeddings for a given audio file."""
    waveform, sample_rate = torchaudio.load(audio_path)
    embeddings = model.encode_batch(waveform).squeeze(0).detach().numpy().flatten()
    return embeddings


# ---------------------------
# Evaluation Metrics
# ---------------------------

def compute_cosine_similarity(reference_path, cloned_path):
    """Compute cosine similarity between reference and cloned audio embeddings."""
    verification_model = get_verification_model()
    reference_embedding = get_audio_embeddings(reference_path, verification_model)
    cloned_embedding = get_audio_embeddings(cloned_path, verification_model)
    return 1 - cosine(reference_embedding, cloned_embedding)

def compute_mfcc_similarity(reference_path, cloned_path):
    """Compute MFCC similarity between reference and cloned audio."""
    ref_audio, ref_sr = librosa.load(reference_path, sr=None)
    cloned_audio, cloned_sr = librosa.load(cloned_path, sr=None)
    ref_mfcc = librosa.feature.mfcc(y=ref_audio, sr=ref_sr, n_mfcc=13)
    cloned_mfcc = librosa.feature.mfcc(y=cloned_audio, sr=cloned_sr, n_mfcc=13)
    ref_mfcc_mean = np.mean(ref_mfcc, axis=1)
    cloned_mfcc_mean = np.mean(cloned_mfcc, axis=1)
    mfcc_distance = euclidean(ref_mfcc_mean, cloned_mfcc_mean)
    return 1 / (1 + mfcc_distance)

def compute_mos(audio_path):
    """Compute MOS (Mean Opinion Score) for an audio file."""
    model = get_mosnet_model()
    mos_score = model.predict(audio_path)
    print(f"MOS Score for {audio_path}: {mos_score:.3f}")
    return mos_score


def evaluate_cloned_audio(reference_path, cloned_path):
    """Evaluate cloned audio using multiple metrics."""
    cosine_similarity = compute_cosine_similarity(reference_path, cloned_path)
    print(f"Cosine Similarity: {cosine_similarity:.3f}")
    mfcc_similarity = compute_mfcc_similarity(reference_path, cloned_path)
    print(f"MFCC Similarity: {mfcc_similarity:.3f}")
    mos_score = compute_mos(cloned_path)
    print(f"MOS Score: {mos_score:.3f}")
    return (0.4 * mfcc_similarity) + (0.4 * cosine_similarity) + (0.2 * mos_score / 5)

# ---------------------------
# Reward Calculation
# ---------------------------

def get_clone_rewards(self, clip_audio_path: str, clone_text: str, responses: List) -> List[float]:
    """Evaluate miner responses and calculate rewards."""
    print("Evaluating cloned audio...==================")
    rewards = []
    for response in responses:
        if not response.clone_audio:
            print("No cloned audio received.=================")
            rewards.append(0.0)
            continue
        print("Cloned audio received.=====================")
        # Generate a unique timestamp-based filename
        timestamp = int(time.time() * 1000)  # Current time in milliseconds
        clone_audio_path = os.path.join(CLONED_AUDIO_DIR, f"cloned_audio_{timestamp}.wav")
        
        # Save the cloned audio to disk
        with open(clone_audio_path, "wb") as audio_file:
            audio_file.write(base64.b64decode(response.clone_audio))
        
        # Transcribe the cloned audio and calculate text correctness
        transcription = transcribe_with_whisper(clone_audio_path)
        print(f"here's  the transcription: {transcription}")
        text_correctness_score = overall_correctness_score(clone_text, transcription)
        print(f"Text Correctness Score: {text_correctness_score}")

        # Skip evaluation if transcription score is too low
        if text_correctness_score < 0.8:
            rewards.append(0.0)
            continue
        
        # Evaluate the cloned audio and calculate final reward
        reward = evaluate_cloned_audio(clip_audio_path, clone_audio_path)
        rewards.append(reward)

    return rewards