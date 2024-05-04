import time
import asyncio
import torch
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Config, Wav2Vec2ForCTC, Wav2Vec2Processor
import os
import glob
import torchaudio
import bittensor as bt
from torch.utils.data import Dataset
import tensorflow as tf
import multiprocessing
import re
from torch.nn.utils.rnn import pad_sequence
import torchaudio.transforms as T
import random
from huggingface_hub import HfApi, upload_file, HfFolder, update_repo_visibility
from healthcare.utils.chain import Chain
from dotenv import load_dotenv
load_dotenv()

@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(os.devnull, 'w') as fnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = fnull, fnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

class AudioDataset(Dataset):
    def __init__(self, audio_paths, transcripts, processor, augmentation_prob=0.5):
        """
        audio_paths: List of paths to audio files.
        transcripts: Corresponding transcriptions for the audio files.
        processor: Wav2Vec2Processor instance for processing audio and text.
        augmentation_prob: Probability of applying augmentation to any given audio file.
        """
        self.audio_paths = audio_paths
        self.transcripts = transcripts
        self.processor = processor
        self.augmentation_prob = augmentation_prob

    def __len__(self):
        """
        Return the number of items in the dataset.
        """
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        waveform, sample_rate = torchaudio.load(self.audio_paths[idx])

        # Randomly decide whether to apply augmentation
        if random.random() < self.augmentation_prob:
            waveform = apply_augmentation(waveform, sample_rate)

        input_values = self.processor(waveform, sampling_rate=sample_rate).input_values[0]
        labels = self.processor(text=self.transcripts[idx]).input_ids

        return {"input_values": torch.tensor(input_values), "labels": torch.tensor(labels)}

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.training_mode = config.training_mode.lower()
        
        if torch.cuda.is_available() and self.config.device.startswith('gpu'):
            # Find device numbers from string
            device_number = 0  # default GPU index
            try:
                numbers_part = self.config.device.split(":")
                device_number = int(re.findall(r'\d+', numbers_part[1])[0])
            except Exception as e:
                print("Error parsing GPU device number, defaulting to 0.")

            self.device = torch.device(f'cuda:{device_number}')
            print(f"Training on GPU: {device_number}")
        else:
            self.device = torch.device('cpu')
            print("Training on CPU")
            
        model_path = 'transcription/miner/model_checkpoints/english'
        self.model, self.processor = self.load_model_and_processor(model_path)        
        self.model.to(self.device)

    def save_model_and_processor(self, save_path):
        """Save the model checkpoint, replacing the previous one."""
        model_save_path = os.path.join(save_path, "current_checkpoint.pt")
        torch.save(self.model.state_dict(), model_save_path)
        bt.logging.info(f"Model saved to {model_save_path}")
        self.processor.save_pretrained(save_path)
        bt.logging.info(f"Processor saved to {save_path}")
    
    @staticmethod
    def load_model_and_processor(model_path):
        model_file_path = f"{model_path}/current_checkpoint.pt"
        processor_directory_path = model_path
        
        if os.path.isfile(model_file_path):
            model = Wav2Vec2ForCTC.from_pretrained(None, state_dict=torch.load(model_file_path), config=Wav2Vec2Config())
            bt.logging.info("Loaded model from checkpoint.")
        else:
            model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
            bt.logging.info("Loaded pretrained model for training.")
        
        processor_files = ['preprocessor_config.json', 'special_tokens_map.json', 'tokenizer_config.json', 'vocab.json']
        if all(os.path.isfile(os.path.join(processor_directory_path, file)) for file in processor_files):
            processor = Wav2Vec2Processor.from_pretrained(processor_directory_path)
            bt.logging.info("Loaded processor from provided files.")
        else:
            processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            bt.logging.info("Loaded pretrained processor for training.")
        
        return model, processor
        
    def train(self):
        if self.config.device.startswith('cpu'):
            # Do not allow gpus
            tf.config.set_visible_devices([], 'GPU')

            # Set the number of cpu cores
            num_cpu_cores = multiprocessing.cpu_count()
            
            try:
                # Find device numbers from string
                numbers_part = self.config.device.split(":")
                num_cores_to_use = min(num_cpu_cores, int(numbers_part[1]))
            except Exception as e:
                num_cores_to_use = num_cpu_cores

            # Set TensorFlow's parallelism threads
            tf.config.threading.set_intra_op_parallelism_threads(num_cores_to_use)
            tf.config.threading.set_inter_op_parallelism_threads(num_cores_to_use)

        elif self.config.device.startswith('gpu'):
            # Find all avaiable gpus
            gpus = tf.config.experimental.list_physical_devices('GPU')

            try:
                # Find device numbers from string
                numbers_part = self.config.device.split(":")
                numbers = re.findall(r'\d+', numbers_part[1])
                device_numbers = [int(num) for num in numbers if int(num) < len(gpus)]
            except Exception as e:
                device_numbers = []

            if not device_numbers:
                device_numbers = [i for i in range(len(gpus))]

            # Set gpus to use
            if gpus:
                try:
                    selected_gpus = [gpus[i] for i in device_numbers if i < len(gpus)]

                    if selected_gpus:
                        tf.config.experimental.set_visible_devices(selected_gpus, 'GPU')

                        for gpu in selected_gpus:
                            tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    selected_gpus = []
        audio_paths, transcripts = self.load_dataset()
        dataset = AudioDataset(audio_paths, transcripts, self.processor)
        if len(dataset) == 0:
            raise ValueError("The dataset is empty. Check data loading and processing.")
        data_loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True, collate_fn=self.collate_batch)
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)

        epoch = 1
        min_loss = float('inf')
        
        if self.config.num_epochs == -1:
            save_path = 'transcription/miner/model_checkpoints/english'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            
            while True:
                total_loss = 0
                num_batches = 0
                
                for batch_idx, batch_data in enumerate(data_loader):
                    input_values = batch_data['input_values'].to(self.device)
                    labels = batch_data['labels'].to(self.device)
                    
                    # Check for NaN in inputs and labels
                    if torch.isinf(input_values).any() or torch.isinf(labels).any():
                        print(f"Infinite values detected in inputs or labels for batch {batch_idx}")
                        continue

                    if torch.isnan(input_values).any():
                        print(f"NaN detected in input_values for batch {batch_idx}")
                        continue  # Skip this batch
                    if torch.isnan(labels).any():
                        print(f"NaN detected in labels for batch {batch_idx}")
                        continue  # Skip this batch
                    
                    optimizer.zero_grad()
                    outputs = self.model(input_values, labels=labels)
                    if torch.isnan(outputs.logits).any():
                        print(f"NaN detected in model outputs for batch {batch_idx}")
                        continue
                    
                    loss = outputs.loss
                    if torch.isnan(loss).any():
                        print(f"NaN detected in loss for batch {batch_idx}")
                        continue  # Skip the backward pass for this batch
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()

                    total_loss += loss.item()
                    num_batches += 1
                    
                    bt.logging.info(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}")
                    
                epoch_loss = total_loss / num_batches
                
                if epoch_loss < min_loss:
                    self.save_model_and_processor(save_path)
                    min_loss = epoch_loss  # Update minimum loss

                    # upload the model metadata HF and write them on-chain
                    # check HF keys
                    access_token = os.getenv("HF_ACCESS_TOKEN")
                    try:
                        api = HfApi()
                        username = api.whoami(access_token)["name"]
                        self.repo_id = username + "/" + os.getenv('REPO_ID')
                        api.create_repo(token=access_token, repo_id=self.repo_id, exist_ok = True)
                    except Exception as e:
                        bt.logging.error(f"❌ Error occured while creating a repository : {e}")
                    
                    HfFolder.save_token(self.access_token)
                    try:
                        # Make the repository as private before uploading
                        update_repo_visibility(self.repo_id, private = True)

                        # Upload the model to hugging face
                        for root, dirs, files in os.walk(self.model_directory):
                            for file in files:
                                # Generate the full path and then remove the base directory part
                                full_path = os.path.join(root, file)
                                relative_path = os.path.relpath(full_path, self.model_directory)
                                with suppress_stdout_stderr():
                                    upload_file(
                                        path_or_fileobj=full_path,
                                        path_in_repo=relative_path,
                                        repo_id=self.repo_id
                                    )
                        bt.logging.info(f"✅ Best model uploaded at {self.repo_id}")

                        # Retrieve the latest commit information
                        api = HfApi()
                        repo_info = api.repo_info(repo_id=self.repo_id, token=self.access_token)
                        last_commit_hash = repo_info.sha
                        
                        # Push the metadata to the chain
                        data = " ".join([self.repo_id, last_commit_hash])
                        while True:
                            try:
                                asyncio.run(self.chain.store_metadata(data))
                                bt.logging.info("✅ Stored the model to the chain.")
                                break
                            except Exception as e:
                                bt.logging.info(f"❌ Error occured while pushing the model to chain : {e}")
                                time.sleep(12)
                                continue

                    except Exception as e:
                        print(f"❌ Error occured while pushing the model : {e}")
                    
                    # Make the repository as public
                    update_repo_visibility(self.repo_id, private = False)

                epoch += 1  # Increment epoch after each complete pass through the data_loader


    def collate_batch(self, batch):
        # Separate input values and labels
        input_values_list = [item['input_values'] for item in batch]
        input_values_list = [torch.squeeze(input_val, 0) for input_val in input_values_list]

        labels_list = [item['labels'] for item in batch]

        # Padding
        input_values_padded = pad_sequence(input_values_list, batch_first=True, padding_value=0.0)
        labels_padded = pad_sequence(labels_list, batch_first=True, padding_value=-100)

        return {'input_values': input_values_padded, 'labels': labels_padded}
    
    def load_dataset(self, base_path='librispeech_dataset'):
        audio_paths = []
        transcripts = []

        # Assuming the LibriSpeech dataset structure
        for subset in ['train-clean-100', 'train-clean-360', 'train-other-500', 'dev-clean', 'dev-other', 'test-clean', 'test-other']:
            subset_path = os.path.join(base_path, 'LibriSpeech', subset)
            for speaker_path in glob.glob(os.path.join(subset_path, '*/')):
                for chapter_path in glob.glob(os.path.join(speaker_path, '*/')):
                    chapter_dir = chapter_path.rstrip('/')
                    path_parts = chapter_dir.split(os.sep)
                    speaker_id = path_parts[-2] 
                    chapter_id = path_parts[-1] 
                    transcript_filename = f"{speaker_id}-{chapter_id}.trans.txt"
                    transcript_path = os.path.join(chapter_dir, transcript_filename)
                    if not os.path.exists(transcript_path):
                        continue
                    with open(transcript_path, 'r') as file:
                        for line in file:
                            line = line.strip()
                            audio_file, transcript = line.split(' ', 1)
                            audio_paths.append(os.path.join(chapter_path, f"{audio_file}.flac"))
                            transcripts.append(transcript)

        return audio_paths, transcripts

def apply_augmentation(waveform, sample_rate):
    augmentation_type = random.choice(['pitch_shift', 'speed_change', 'add_noise', 'none'])

    if augmentation_type == 'pitch_shift':
        n_steps = random.randint(-2, 2)  # Similar range as the validator
        pitch_shift = n_steps * 100  # Convert semitones to cents for sox
        waveform, _ = torchaudio.sox_effects.apply_effects_tensor(waveform, sample_rate, [['pitch', str(pitch_shift)]])

    elif augmentation_type == 'speed_change':
        speed_factor = random.uniform(0.9, 1.1)  # Similar range as the validator
        waveform, _ = torchaudio.sox_effects.apply_effects_tensor(waveform, sample_rate, [['speed', str(speed_factor)]])

    elif augmentation_type == 'add_noise':
        noise_intensity = random.uniform(0.001, 0.005)  # Adjust based on validator's noise level
        noise = torch.randn(waveform.size()) * noise_intensity
        waveform += noise

    return waveform