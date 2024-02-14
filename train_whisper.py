import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from whisper.model import Whisper, ModelDimensions
from torch.nn.utils.rnn import pad_sequence
import whisper
import os
import torchaudio

# Set the audio backend for torchaudio to "soundfile"
torchaudio.set_audio_backend("soundfile")

# Set the device for computation (GPU if available, otherwise CPU)
DEVICE = "cuda"

# Hyperparameters
batch_size = 1
num_workers = 1
learning_rate = 0.005

# Dataset class for LibriSpeech
class LibriSpeech(torch.utils.data.Dataset):
    def __init__(self, split="test-clean", device="cuda"):
        # Initialize LibriSpeech dataset from torchaudio
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root=os.path.expanduser("LibriSpeech"),
            url=split,
            download=True,
        )
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        # Retrieve audio, sample rate, text, and other information from the dataset
        audio, sample_rate, text, _, _, _ = self.dataset[item]
        assert sample_rate == 16000

        # Preprocess audio data using Whisper functions
        audio = whisper.pad_or_trim(audio.flatten()).to(self.device)
        mel = whisper.log_mel_spectrogram(audio)

        # Return dictionary containing mel spectrogram and text tokens
        return {'mel': mel, 'tokens': text}

# Loss function
def compute_loss(logits, targets):
    return nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

# Collate function for DataLoader
def collate_fn(batch):
    mel_list = [item['mel'] for item in batch]
    tokens_list = [item['tokens'] for item in batch]
    mel_tensor = torch.stack(mel_list)
    return mel_tensor, tokens_list

# Training function for a single epoch
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0

    # Iterate over batches in the DataLoader
    for mel, tokens_list in tqdm(dataloader, desc="Training", leave=False):
        for tokens in tokens_list:

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass through the model
            output = model(mel, tokens)

            # Compute the loss and backpropagate
            loss = compute_loss(output, tokens)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    return total_loss / len(dataloader)

# Function to train the Whisper model
def train_whisper():
    # Initialize LibriSpeech dataset and DataLoader
    dataset = LibriSpeech()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)

    # Model dimensions for Whisper
    dims = ModelDimensions(n_mels=80, n_audio_ctx=512,
        n_audio_state=512, n_audio_head=8,
        n_audio_layer=6, n_vocab=5000,
        n_text_ctx=512, n_text_state=512,
        n_text_head=8, n_text_layer=6,
    )

    # Instantiate Whisper model and optimizer
    whisper_model = Whisper(dims).to(DEVICE)
    optimizer = Adam(whisper_model.parameters(), lr=learning_rate)

    # Number of training epochs
    num_epochs = 10 

    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = train_epoch(whisper_model, dataloader, optimizer, DEVICE)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    # Save the trained model
    torch.save(whisper_model.state_dict(), 'whisper_model.pth')

# Entry point for the script
if __name__ == '__main__':
    train_whisper()
