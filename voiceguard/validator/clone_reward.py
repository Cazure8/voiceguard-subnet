import os
import time
import base64
import torchaudio
import numpy as np
import librosa
from pydub import AudioSegment
from typing import List
from scipy.spatial.distance import cosine, euclidean
from speechbrain.inference import SpeakerRecognition
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
        VERIFICATION_MODEL = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models_speechbrain")
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


# ---------------------------
# Evaluation Metrics
# ---------------------------

def compute_cosine_similarity(ref_file, cloned_file):
    # Load pre-trained model
    verification = get_verification_model()
    
    # Load audio files and their sampling rates
    ref_audio, ref_sr = torchaudio.load(ref_file)
    cloned_audio, cloned_sr = torchaudio.load(cloned_file)
    
    # Ensure both audio files have the same sampling rate
    if ref_sr != 16000:
        ref_audio = torchaudio.transforms.Resample(orig_freq=ref_sr, new_freq=16000)(ref_audio)
    if cloned_sr != 16000:
        cloned_audio = torchaudio.transforms.Resample(orig_freq=cloned_sr, new_freq=16000)(cloned_audio)

    # Get speaker embeddings
    ref_embedding = verification.encode_batch(ref_audio)
    cloned_embedding = verification.encode_batch(cloned_audio)
    
    # Compute similarity score
    similarity = verification.similarity(ref_embedding, cloned_embedding)
    return similarity.item()

# def compute_mfcc_similarity(reference_path, cloned_path):
#     """Compute MFCC similarity between reference and cloned audio."""
#     ref_audio, ref_sr = librosa.load(reference_path, sr=None)
#     cloned_audio, cloned_sr = librosa.load(cloned_path, sr=None)
#     ref_mfcc = librosa.feature.mfcc(y=ref_audio, sr=ref_sr, n_mfcc=13)
#     cloned_mfcc = librosa.feature.mfcc(y=cloned_audio, sr=cloned_sr, n_mfcc=13)
#     ref_mfcc_mean = np.mean(ref_mfcc, axis=1)
#     cloned_mfcc_mean = np.mean(cloned_mfcc, axis=1)
#     mfcc_distance = euclidean(ref_mfcc_mean, cloned_mfcc_mean)
#     return 1 / (1 + mfcc_distance)

def analyze_pitch(audio_path, sr=None, f0_min=50.0, f0_max=300.0):
    """
    Loads an audio file and computes the mean pitch (F0 in Hz) using librosa's pyin.
    Returns None if no voiced frames are found.
    
    :param audio_path: Path to the audio file.
    :param sr: Sample rate to use when loading (None = use the file's native rate).
    :param f0_min: Minimum expected F0 (Hz) for pitch tracking.
    :param f0_max: Maximum expected F0 (Hz) for pitch tracking.
    :return: Mean pitch (float) in Hz, or None if no voiced frames are detected.
    """
    y, sr = librosa.load(audio_path, sr=None)
    f0, voiced_flags, voiced_prob = librosa.pyin(y, sr=sr, fmin=f0_min, fmax=f0_max)
    f0_voiced = f0[~np.isnan(f0)]
    if len(f0_voiced) > 0:
        return float(np.mean(f0_voiced))
    else:
        return None

def compare_mean_pitch(ref_audio_path, cloned_audio_path, sr=None, f0_min=10.0, f0_max=500.0):
    """
    Computes the difference in mean pitch (Hz) between two audio files.

    :param ref_audio_path: Path to the reference audio file
    :param cloned_audio_path: Path to the cloned audio file
    :param sr: Sample rate to use when loading (None = use file's native rate)
    :param f0_min: Minimum expected F0 (Hz) for pitch tracking
    :param f0_max: Maximum expected F0 (Hz) for pitch tracking
    :return: Absolute difference in mean pitch (float) if both have voiced frames,
             otherwise None
    """
    ref_mean_pitch = analyze_pitch(ref_audio_path, sr=sr, f0_min=f0_min, f0_max=f0_max)
    cloned_mean_pitch = analyze_pitch(cloned_audio_path, sr=sr, f0_min=f0_min, f0_max=f0_max)

    if ref_mean_pitch is not None and cloned_mean_pitch is not None:
        return abs(ref_mean_pitch - cloned_mean_pitch)
    else:
        return None

def pitch_diff_to_similarity(diff):
    """
    Convert a non-negative pitch difference (in Hz) to a 0.0–1.0 similarity score,
    using the formula: similarity = 1 / (1 + diff).

    - If diff = 0, similarity = 1.0 (perfect).
    - If diff is very large, similarity approaches 0.0.

    :param diff: Non-negative pitch difference in Hz (float), or None.
    :return: Similarity in [0.0, 1.0], or None if diff is None.
    """
    if diff is None:
        return None
    diff = max(0.0, diff)
    return 1.0 / (1.0 + diff)

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
    difference = compare_mean_pitch(reference_path, cloned_path)
    pitch_similarity = pitch_diff_to_similarity(difference)    
    print(f"F0 Similarity Score: {pitch_similarity:.3f}")
    mos_score = compute_mos(cloned_path)
    print(f"MOS Score: {mos_score:.3f}")
    return (0.25 * pitch_similarity) + (0.5 * cosine_similarity) + (0.25 * mos_score / 5)

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
        print(f"Transcribing cloned audio: {clone_audio_path}")
        transcription = transcribe_with_whisper(clone_audio_path)
        print(f"tttttttranscription: {transcription}")
        print(f"ccccccclone_text: {clone_text}")
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