<div align="center">

# **Transcription Subnet** <!-- omit in toc -->
[![Discord Chat](https://img.shields.io/discord/308323056592486420.svg)](https://discord.gg/bittensor)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

---

## The Incentivized Internet <!-- omit in toc -->

[Discord](https://discord.gg/bittensor) • [Network](https://taostats.io/) • [Research](https://bittensor.com/whitepaper)
</div>

---
- [Introduction](#introduction)
- [Installation](#installation)
- [Running](#running)
  - [Running subtensor locally](#before-you-proceed)
  - [Running miner](#running-miner)
  - [Running validator](#running-validator)
- [Writing your own incentive mechanism](#writing-your-own-incentive-mechanism)
- [License](#license)

---

## Introduction

**IMPORTANT**: If you are new to Bittensor subnets, read this section before proceeding to [Installation](#installation) section. 

The Bittensor blockchain hosts multiple self-contained incentive mechanisms called **subnets**. Subnets are playing fields in which:
- Subnet miners who produce value, and
- Subnet validators who produce consensus

---

## Installation

#### Bittensor

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/opentensor/bittensor/master/scripts/install.sh)"
```

#### Clone the repository from Github
```bash
git clone https://github.com/Cazure8/transcription-subnet.git
```

#### Install package dependencies for the repository
```bash
cd transcription-subnet
apt install python3-pip -y
python3 -m pip install -e .
```

#### Install `pm2`
```bash
apt update && apt upgrade -y
apt install nodejs npm -y
npm i -g pm2
```
---

## Running

### Running subtensor locally

#### Install Docker
```bash
apt install docker.io -y
apt install docker-compose -y
```

#### Run Subtensor locally
```bash
git clone https://github.com/opentensor/subtensor.git
cd subtensor
docker-compose up --detach
```

### Running miner
In the innovative transcription subnet, miners play a vital role in transcribing spoken language into written text. Initially utilizing Google's Speech-to-Text API, these miners focus on delivering fast and accurate transcriptions. Their contributions are essential in making audio content accessible and searchable, greatly enhancing the utility of spoken information across various domains.

Looking ahead, the transcription subnet plans to transit custom transcription model soon, offering an alternative to Google's API. This shift aims to provide miners with a more customizable and potentially more cost-effective solution, empowering them to further optimize transcription accuracy and efficiency. As the subnet evolves, miners who adeptly adapt to and leverage DeepSpeech's capabilities will likely see enhanced rewards, reflecting the subnet's commitment to continual innovation and excellence in transcription services.

Miners should have Google's Speech-to-Text API key to be able to provide transcription.

#### Getting Google's Speech-to-Text API key

To utilize Google's Speech-to-Text service in the initial phase of our transcription subnet, you will need to obtain an API key from Google Cloud. This key allows you to access Google's powerful speech recognition capabilities. Here’s a simple guide to getting your API key:

Please take a reference from images in [docs/get_google_cloud_credential](docs/get_google_cloud_credential)

- Create a Google Cloud Account: If you don't already have one, sign up for a Google Cloud account at cloud.google.com. You can simply go to https://console.cloud.google.com/welcome

- Create a New Project: Once logged in, create a new project from the Google Cloud Console.

- Enable Speech-to-Text API: Navigate to the "API & Services" dashboard, search for the Speech-to-Text API, and enable it for your project.

- Set up Billing: To use the Speech-to-Text API, you must set up billing with Google Cloud. Note that Google often offers a free trial with credits that can be used for their APIs. Setting up billing should be with Credit Card.

- Create Credentials: In the API & Services dashboard, go to "Credentials" and create a new API key. This key will be used in our transcription subnet.

- Secure Your API Key: Store this key securely and do not share it publicly, as it can be used to access Google Cloud services on your behalf.


#### Future Plans: Transitioning to a Custom Public Audio-to-Speech Model
We understand the reliance on Google's API might not align with the long-term vision of all our users. To address this, we're excited to announce plans to introduce a custom, public audio-to-speech model soon. This shift aims to provide a more open, adaptable, and potentially cost-effective solution for transcription.

We're committed to ensuring a seamless transition to this new model and will provide ample support and resources as we make this exciting leap forward. We appreciate your understanding and patience during this phase of growth and innovation.


#### Run the miner with `pm2`

```bash
# To run the miner
pm2 start neurons/miner.py --interpreter python3 -- --netuid 11 --subtensor.network <LOCAL/FINNEY/TEST> --wallet.name <WALLET NAME> --wallet.hotkey <HOTKEY NAME> --axon.port <PORT>
```

### Running validator

Validators are essential to the integrity and efficiency of our transcription subnet. They are responsible for assessing the quality and accuracy of the transcriptions provided by miners. Here’s an overview of their crucial role:

Validators frequently dispatch a variety of audio clips to miners, covering a wide range of languages, dialects, and audio qualities. This diverse array of samples ensures that miners are adept at handling a broad spectrum of transcription challenges. Validators then meticulously evaluate the transcriptions returned by miners, focusing on accuracy, speed, and adherence to context.

The scoring process by validators is rigorous and fair, aiming to objectively assess each miner's performance. This evaluation is not just about the literal accuracy of the transcriptions, but also about understanding the nuances of spoken language and context. Validators contribute significantly to the continuous improvement of transcription models, driving the entire subnet towards excellence.

In the transcription subnet, validators thus uphold the highest standards of performance. Their diligent work ensures that the subnet remains a reliable and authoritative source for converting audio content into accurate text, thereby enhancing the overall value and usability of spoken data.

#### Download the dataset for transcriptin scoring
```bash
python3 -m spacy download en_core_web_md
```

```bash
# To run the validator
pm2 start ./validators/validator.py --interpreter python3 -- --netuid 11 --subtensor.network <LOCAL/FINNEY/TEST> --wallet.name <WALLET NAME> --wallet.hotkey <HOTKEY NAME>
```

## License
This repository is licensed under the MIT License.
```text
# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

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
