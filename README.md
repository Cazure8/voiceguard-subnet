<div align="center">

# **Voiceguard Subnet** <!-- omit in toc -->
![Subnet]()



[![Discord Chat](https://img.shields.io/discord/308323056592486420.svg)](https://discord.com/channels/799672011265015819/1161765231953989712)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

---

### Decentralized Audio-Deepfake Detection AI on the Bittensor Network<!-- omit in toc -->

[Discord](https://discord.com/channels/799672011265015819/1161765231953989712)  •  [Subnet Status](https://x.taostats.io/subnet/44)
</div>


---
- [Introduction](#introduction)
- [Key Features](#key-features)
- [System Design](#system-design)
  - [Miner](#miner)
  - [Validator](#validator)
- [Installation](#installation)
- [Running](#running)
  - [Running subtensor locally](#before-you-proceed)
  - [Running miner](#running-miner)
  - [Running validator](#running-validator)
- [Roadmap](#roadmap)
- [License](#license)

---
## Introduction

Voiceguard Subnet is a **groundbreaking decentralized audio-AI** subnet built on the Bittensor network. Our platform revolutionizes voice cloning, deepfake detection, and high-speed speech-to-text (STT) services, while creating specialized datasets to advance AI audio research. By harnessing the decentralization power of Bittensor, Voiceguard enables scalable, transparent, and collaborative innovation in the audio AI industry.

## Key Features

-	🎯 **Advanced Deepfake Detection** Detect AI-generated voices with pioneering accuracy using our state-of-the-art deepfake detection model.
- 🎙️ **Voice Cloning** Leverage cutting-edge AI to clone voices efficiently in multi-languages. 
- 📝 **Fast & Accurate Speech-to-Text** Offers multi-language, distributed transcription services optimized for speed and precision.
- 📚 **Specialized Audio Dataset Creation** Build vast, high-quality datasets of real and AI-generated voices for training voice AI models.

## System Design

### Synapse Types

 Voiceguard validates miner responses across three key synapses:
- **Type I (STT):** Speech-to-text conversion for YouTube videos.
- **Type II (Clone):** Generate cloned audio based on voice samples and text.
- **Type III (Detection):** Detect whether a given audio is real or AI-generated.

### Miner

 Miners are tasked to:
-	Respond to all **three synapse types** effectively.
-	Run **Whisper Large model** to ensure accurate STT services.
-	Implement AI models to handle voice cloning and deepfake detection tasks.

### Validator

Validators send requests equally among the three synapse types and reward miners as follows:

-	Type I (STT): 20%.
-	Type II (Voice Clone): 40%.
-	Type III (Deepfake Detection): 40%.


## Installation

### System Requirements

-	Python Version: 3.10+
-	Hardware: Dedicated GPU (16GB+), 1Gbps Network Speed
-	Environment: Use a virtual environment for dependency management.

### Setup

1.	Install Bittensor
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/opentensor/bittensor/master/scripts/install.sh)"
```
2.	Clone the Repository
```bash
git clone https://github.com/Cazure8/voiceguard-subnet
cd voiceguard-subnet
```
3.	Install Dependencies
```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
sudo apt-get update && sudo apt-get install ffmpeg
```
4.	Download Necessary Datasets
```bash
python3 voiceguard/utils/download.py
```
5.	Install PM2 (Optional for Process Management)
```bash
apt update && apt upgrade -y
apt install nodejs npm -y
npm i -g pm2
```
6.	Set Up Proxy (Recommended for YouTube Rate Limits)
Configure your proxy URL in the .env file.

## Running

### Running the Miner
```bash
 # To run the miner
pm2 start neurons/miner.py --name miner --interpreter python3 -- 
    --netuid # the subnet netuid, default = 
    --subtensor.network # the bittensor chain endpoint, default = finney, local, test (highly recommend running subtensor locally)
    --wallet.name # your wallet coldkey name, default = default
    --wallet.hotkey # your wallet hotkey name, default = default
    --axon.ip # your IP
    --axon.port # the port you allowed
    --logging.debug # run in debug mode, alternatively --logging.trace for trace mode
```

```bash
# simple Example
pm2 start neurons/miner.py --interpreter python3 -- --netuid 253 --wallet.name myminer --wallet.hotkey myhotkey
```

### Running the Validator
```bash
# To run the validator
pm2 start neurons/validator.py --interpreter python3 -- --netuid 44 --subtensor.network <LOCAL/FINNEY/TEST> --wallet.name <WALLET NAME> --wallet.hotkey <HOTKEY NAME> --axon.ip <YOUR IP> --axon.port <YOUR PORT>
```

```bash
# simple Example
pm2 start neurons/validator.py --interpreter python3 -- --netuid 44 --wallet.name myvalidator --wallet.hotkey myhotkey
```

## Roadmap

**Q1, 2025**

-	✅ Testnet launch: Testnet UID 253
-	🔄 Prepare APIs and web applications for launch.
-	⚙️ Obtain community validation of logic.
-	🚀 Launch on Bittensor mainnet.
-	📊 Build a robust real and fake voice dataset with 100+ hours.

**Q2, 2025**

-	📈 Expand datasets to 500+ hours.
-	🌐 Enable multi-language, any-audio transcription for APIs.
-	🛠 Develop Chrome extension and iOS app for services.

**Q3, 2025**

-	⏱ Roll out real-time API services.
-	🎥 Partner with the film industry for voice cloning services.
-	🎯 Provide 3,000+ hours of real and cloned datasets.


## License
This repository is licensed under the MIT License.
```text
# The MIT License (MIT)
# Copyright © 2025 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
```
