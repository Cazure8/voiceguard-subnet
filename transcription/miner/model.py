import torch
from torch.utils.data import DataLoader
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import os
import glob
import torchaudio
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, audio_paths, transcripts, processor):
        """
        audio_paths: List of paths to audio files.
        transcripts: List of transcriptions corresponding to the audio files.
        processor: Wav2Vec2 processor for processing both audio and text data.
        """
        self.audio_paths = audio_paths
        self.transcripts = transcripts
        self.processor = processor

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        # Load audio
        waveform, sample_rate = torchaudio.load(self.audio_paths[idx])
        
        # Process waveform with processor
        input_values = self.processor(waveform, sampling_rate=sample_rate).input_values[0]

        # Encode the transcript to get label ids
        with self.processor.as_target_processor():
            labels = self.processor(self.transcripts[idx]).input_ids

        return {"input_values": input_values, "labels": labels}

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.training_mode = config.training_mode.lower()
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(self.device)
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    def train(self):
        audio_paths, transcripts = self.load_dataset()
        dataset = AudioDataset(audio_paths, transcripts, self.processor)
        data_loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)

        for epoch in range(self.config.num_epochs):
            for batch_idx, (input_values, labels) in enumerate(data_loader):
                input_values = input_values.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(input_values, labels=labels)
                loss = outputs.loss

                loss.backward()
                optimizer.step()

                print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}")


    def load_dataset(self, base_path='librispeech_dataset'):
        audio_paths = []
        transcripts = []

        # Assuming the LibriSpeech dataset structure
        for subset in ['train-clean-100', 'train-clean-360', 'train-other-500', 'dev-clean', 'dev-other', 'test-clean', 'test-other']:
            subset_path = os.path.join(base_path, subset)
            for speaker_path in glob.glob(os.path.join(subset_path, '*/*')):
                for chapter_path in glob.glob(os.path.join(speaker_path, '*')):
                    transcript_path = os.path.join(chapter_path, f"{os.path.basename(chapter_path)}.trans.txt")
                    with open(transcript_path, 'r') as file:
                        for line in file:
                            line = line.strip()
                            audio_file, transcript = line.split(' ', 1)
                            audio_paths.append(os.path.join(chapter_path, f"{audio_file}.flac"))
                            transcripts.append(transcript)

        return audio_paths, transcripts

    