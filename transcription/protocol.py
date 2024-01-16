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
    
    Attributes:
    - audio_input: A bytes-like object representing the audio data or a string URL to the audio file.
    - transcription_output: A string representing the transcribed text, filled by the receiving miner.
    """

    # Required request input, can be raw audio data or an audio URL.
    audio_input: str = ""
    # audio_input: None

    # Optional response output, filled by the receiving miner.
    transcription_output: typing.Optional[str] = ""

    def deserialize(self) -> str:
        """
        Deserialize the transcription output. This method retrieves the transcription response from
        the miner, deserializes it, and returns it as the output of the dendrite.query() call.

        Returns:
        - str: The deserialized transcription response.

        Example:
        >>> transcription_instance = TranscriptionSynapse(audio_input=b'audio data here')
        >>> transcription_instance.transcription_output = 'This is transcribed text.'
        >>> transcription_instance.deserialize()
        'This is transcribed text.'
        """
        return self.transcription_output 

