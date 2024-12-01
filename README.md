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
- [Miner and Validator Functionality](#miner-and-validator-functionality)
  - [Miner](#miner)
  - [Validator](#validator)
- [Installation](#installation)
- [Running](#running)
  - [Running subtensor locally](#before-you-proceed)
  - [Running miner](#running-miner)
  - [Running validator](#running-validator)
- [License](#license)

---
## Introduction

Welcome to the Voiceguard Subnet on the Bittensor network, your cutting-edge solution against voice deepfakes. Voice deepfakes are becoming a more serious problem each day with the rapid advancement of AI technology. Several critical incidents have been reported, such as the deepfake robocall incident in New Hampshire, which used AI to deter voters​ [source](https://www.politico.com/news/2024/02/06/robocalls-fcc-new-hampshire-texas-00139864)​​​, and numerous other significant deepfake incidents documented in various reports​ [source](https://virtualdoers.com/10-notable-deepfake-incidents-in-the-internet/)​.

Voiceguard Subnet aims to resolve these issues through the power of decentralization. This subnet offers VOICE CLONING, robust voice DEEPFAKE DETECTION through and provides high-speed, accurate transcriptions for YouTube videos. Additionally, it offers a specialized database for training voice deepfake detection models. In the audio AI field, preparing datasets has been a significant challenge, but by harnessing Bittensor's decentralization power, we are resolving that problem together with the best model generation for voice cloning and deepfake detection.

## Key Features

- 🧠 **Generate high quality of audio deepfake detection**: Voiceguard subnet will generate a pioneering AI model to distinguish between real voice and AI generated one.
- 🧠 **Voice cloning model**: Advanced voice cloning model and its service will be provided.
- 🧠 **Fast, accurate transcription**: Distributed transcription will provide high-speed, accurate transcription for Youtube videos in any language.
- 🧠 **Large sized special database**: By harnessing miners hard work, specialized datasets for training anti-voice deepfake AI model will be created.

## Miner and Validator Functionality

### Miner

- Every miner is required to operate the Whisper Large model, facilitating a fine-tuned API for database generation and interactive video functions.
- Specialized Contribution: Miners with AI expertise may opt to train advanced detection models using our provided database. They should upload the metadata for trained model to the Hugging Face. The top three performers in this category will share thirty percent of the total rewards. Note: These miners must also maintain the Whisper model to ensure seamless API functionality.

### Validator

- Performance Evaluation: Validators assess miners based on the response time and accuracy of their benchmarking response.
- Model Optimization: They periodically review trained data from Hugging Face and calculate loss metrics to enhance model performance.


## Installation

**Python version should be over 3.10.** <br>
**Using dedicated virtual environment is recommended.**

### Bittensor

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/opentensor/bittensor/master/scripts/install.sh)"
```

#### Clone the repository from Github
```bash
git clone https://github.com/Cazure8/voiceguard-subnet
```

#### Install package dependencies for the repository
```bash
cd voiceguard-subnet
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

#### Install external packages
```bash
sudo apt-get update
sudo apt-get install ffmpeg
sudo apt-get install espeak
sudo apt install libsox-dev
```

#### Download datasets for scoring
```bash
python3 voiceguard/utils/download.py
```

#### Install `pm2`
```bash
apt update && apt upgrade -y
apt install nodejs npm -y
npm i -g pm2
```

#### Using proxy (Optional but strongly recommended)

All validators and miners are strongly recommended to use a proxy to circumvent YouTube's download limitations and ensure reliable vtrust/trust. You have the option to set up your own proxy server. Alternatively, if you prefer using a proxy service, you can obtain a paid version rotating proxy, for example, from SmartProxy Dashboard(https://dashboard.smartproxy.com/).
Get the rotating proxy url and put that it in .env file. Sample file .env.sample is already there in the codebase.

## Running

### Miner

#### Minimum Hardware Requirements for Miners
To participate effectively as a miner in Voiceguard Subnet, your system should meet the following **minimum** requirements:

- **Network Speed**:  **1Gbps**
- **GPU**: **16GB**

#### Run the miner with `pm2` for normal contributors
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
pm2 start neurons/miner.py --interpreter python3 -- --netuid 44 --wallet.name myminer --wallet.hotkey myhotkey
```

#### Run the miner for AI model trainers
```bash
python3 miner_model/upload_model.py --hf_repo_id HF_REPO --wallet.name WALLET  --wallet.hotkey HOTKEY --model_dir PATH_TO_MODEL   
```
### Validator
#### Minimum Hardware Requirements for Validators
To participate effectively as a validator in Voiceguard Subnet, your system should meet the following **minimum** requirements:

- **Network Speed**:  **1Gbps**
- **GPU**: **16GB**

#### Run the validator with `pm2`
```bash
# To run the validator
pm2 start neurons/validator.py --interpreter python3 -- --netuid 44 --subtensor.network <LOCAL/FINNEY/TEST> --wallet.name <WALLET NAME> --wallet.hotkey <HOTKEY NAME> --axon.ip <YOUR IP> --axon.port <YOUR PORT>
```

```bash
# simple Example
pm2 start neurons/validator.py --interpreter python3 -- --netuid 44 --wallet.name myvalidator --wallet.hotkey myhotkey
```

## Roadmap

- [x] Launch Voiceguard subnet on Bittensor testnet
- [ ] Confirm transcription request and response working smoothly, and fully test its API from miners
- [ ] Get approved voice cloning validation logic from community like OTF
- [ ] Lauch Voiceguard subnet on Bittensor mainnet
- [ ] Build real and fake audio datasets for training voice cloning and deepfake detection model
- [ ] Publish web application with Transcribe and Taoscribe based on stable API from miners
- [ ] Start voice cloning competition
- [ ] Reach 500+ hours of Youtube audio clips
- [ ] Start deepfake detection model competition
- [ ] Connect web application with the voice cloning and deepfake detection model API
- [ ] Provide 5000+ hours of real and cloned audio clips
- [ ] Go public for the datasets
- [ ] Upgrade web application to handle any audio transcription
- [ ] Enhance voice deepfake API for various languages


## License
This repository is licensed under the MIT License.
```text
# The MIT License (MIT)
# Copyright © 2024 Yuma Rao

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
