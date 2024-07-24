<div align="center">

# **Voice Guard Subnet** <!-- omit in toc -->
![Subnet44](docs/voiceguard-white.png)



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

Welcome to the Voice Guard Subnet on the Bittensor network, your cutting-edge solution against voice deepfakes. Voice deepfakes are becoming a more serious problem each day with the rapid advancement of AI technology. Several critical incidents have been reported, such as the deepfake robocall incident in New Hampshire, which used AI to deter voters​ [source](https://www.politico.com/news/2024/02/06/robocalls-fcc-new-hampshire-texas-00139864)​​​, and numerous other significant deepfake incidents documented in various reports​ [source](https://virtualdoers.com/10-notable-deepfake-incidents-in-the-internet/)​.

Voice Guard Subnet aims to resolve these issues through the power of decentralization. This subnet offers robust voice fake detection through model fine-tuning competitions and provides high-speed, accurate transcriptions for YouTube videos. Additionally, it offers a specialized database for training voice deepfake detection models. In the audio AI field, preparing datasets has been a significant challenge, but by harnessing Bittensor's decentralization power, we are resolving that problem together with the best model generation.


## Key Features

- 🧠 **Generate high quality of audio deepfake detection**: Voice guard subnet will generate a pioneering AI model to distinguish beween real voice and AI generated one.
- 🧠 **Fast, accurate transcription**: Distributed transcription will provide high-speed, accurate transcription for Youtube videos in any language.
- 🧠 **Large sized special database**: By harnessing miners hard work, specialized datasets for training anti-voice deepfake AI model will be created.
- 🧠 **Fine tune any model in audio field**: Any audio related model can be fine-tuned for specific purpose with least loss value in training.
- 🧠 **Friendly to everyone**: Not only tech people, but also non-tech ones can also get reward propelling Bittensor's evolution.

## Miner and Validator Functionality

### Miner

- General Participation: Every miner is required to operate the Whisper Large model, facilitating a fine-tuned API for database generation and interactive video functions. This is accessible to non-technical individuals with moderate resources. Seventy percent of total rewards are distributed among these contributors.
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

---

<br><br>

## Running

### Miner

#### Minimum Hardware Requirements for Miners
To participate effectively as a miner in Voice Guard Subnet, your system should meet the following **minimum** requirements:

- **Network Speed**:  **1Gbps**
- **GPU**: **20GVRAM**

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
<br>

### Validator
#### Minimum Hardware Requirements for Validators
To participate effectively as a validator in Voice Guard Subnet, your system should meet the following **minimum** requirements:

- **Network Speed**:  **1Gbps**
- **GPU**: **20GVRAM**

#### Run the validator with `pm2`
```bash
# To run the validator
pm2 start neurons/validator.py --interpreter python3 -- --netuid 44 --subtensor.network <LOCAL/FINNEY/TEST> --wallet.name <WALLET NAME> --wallet.hotkey <HOTKEY NAME> --axon.ip <YOUR IP> --axon.port <YOUR PORT>
```

```bash
# simple Example
pm2 start neurons/validator.py --interpreter python3 -- --netuid 44 --wallet.name myvalidator --wallet.hotkey myhotkey
```
<br>

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
