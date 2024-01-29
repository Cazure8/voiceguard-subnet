# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 Cazure

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

import typing
import bittensor as bt

class Transcription(bt.Synapse):
    """
    Protocol representation for handling transcription requests in the transcription subnet.
    Handles both raw audio data and audio URLs.
    """
    # Type indicator: 'data' for raw audio data, 'url' for audio URL
    input_type: str = 'data'
    
    # # Required request input, can be raw audio data or an audio URL.
    # audio_input: str = ""
    
    # Audio input can be raw audio data (bytes) or an audio URL (str).
    # audio_input: typing.Union[bytes, str] = ""
    audio_input: str = ''
    # audio_input: None

    # Optional response output, filled by the receiving miner.
    transcription_output: typing.Optional[str] = ""

    def deserialize(self) -> str:
        return self.transcription_output 
    
    def is_url(self) -> bool:
        """ Check if the input is a URL. """
        return self.input_type == 'url'

