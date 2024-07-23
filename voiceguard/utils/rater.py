import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from torch.utils.data import DataLoader

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, audio_paths, transcripts, processor):
        self.audio_paths = audio_paths
        self.transcripts = transcripts
        self.processor = processor

    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        waveform, sample_rate = torchaudio.load(self.audio_paths[idx])
        input_values = self.processor(waveform, sampling_rate=sample_rate, return_tensors="pt").input_values.squeeze(0)
        labels = self.processor.tokenizer(self.transcripts[idx], return_tensors="pt").input_ids.squeeze(0)
        return {"input_values": input_values, "labels": labels}

def rate(ckpt_path, audio_paths, transcripts, batch_size=16):
    processor = Wav2Vec2Processor.from_pretrained(ckpt_path)
    dataset = AudioDataset(audio_paths, transcripts, processor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = Wav2Vec2ForCTC.from_pretrained(ckpt_path)
    model.eval()

    criterion = torch.nn.CTCLoss(blank=model.config.ctc_config.blank_id)
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            input_values = batch["input_values"]
            labels = batch["labels"]
            logits = model(input_values).logits
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            input_lengths = torch.full((logits.size(0),), logits.size(1), dtype=torch.long)
            label_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)

            loss = criterion(log_probs.transpose(0, 1), labels, input_lengths, label_lengths)
            total_loss += loss.item()
            total_batches += 1

    avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
    return avg_loss
