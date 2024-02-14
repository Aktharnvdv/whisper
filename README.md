# Whisper: Speech Synthesis with Whisper ASR

This repository contains an implementation of training the Whisper Automatic Speech Recognition (ASR) model using the LibriSpeech dataset. The Whisper model is part of the Whisper ASR system, designed for speech synthesis.

## Requirements

- [PyTorch](https://pytorch.org/)
- [torchaudio](https://pytorch.org/audio/stable/index.html)
- [whisper](https://github.com/snakers4/whisper)

# Install the required libraries using the following command:

```bash
pip install torch torchaudio git+https://github.com/snakers4/whisper
```
# Dataset
The code uses the LibriSpeech dataset for training. It automatically downloads the specified split (e.g., "test-clean") and preprocesses the audio data.

Usage
Clone the repository:
```bash
git clone https://github.com/Aktharnvdv/whisper.git
cd whisper
```
Install the required libraries as mentioned above.

Run the training script:

```bash
python train_whisper.py
```
Configuration
You can customize the training configuration, such as the batch size, number of workers, learning rate, and model dimensions, by modifying the corresponding variables at the beginning of the script.

# Model
The Whisper model is initialized with the specified dimensions in the Model Dimensions class. The model is trained using the LibriSpeech dataset for a specified number of epochs.

# Training
The train_whisper function initializes the dataset, data loader, model, and optimizer. It then trains the Whisper model for the specified number of epochs.

# Results
After training, the Whisper model's state dictionary is saved to a file named whisper_model.pth.
