# The MIT License (MIT)
# Copyright © 2025 Cazure

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

class VoiceGuardSynapse(bt.Synapse):
    """
    Protocol representation for handling voiceguard requests and responses in the VoiceGuard subnet.
    
    This protocol supports three types of synapse interactions:
    - Type I (stt): Speech-to-text conversion. Uses a YouTube URL and a time segment to generate transcription.
    - Type II (clone): Voice cloning. Accepts a voice clip and text to generate a cloned audio output.
    - Type III (detection): Deepfake detection. Processes an audio clip and returns a prediction of real or fake.
    """
    # Synapse type: 'stt', 'clone', or 'detection'
    synapse_type: str = 'stt'
    
    # Fields for Type I (Speech-to-Text)
    stt_link: str = ''  # URL for the audio source (YouTube URL)
    stt_segment: typing.Optional[typing.Tuple[int, int]] = None  # Start and end times in seconds
    stt_transcription: typing.Optional[str] = None  # Transcription output

    # Fields for Type II (Voice Cloning)
    clone_clip: typing.Optional[str] = None  # Audio clip data as bytes
    clone_text: typing.Optional[str] = None  # Text to be synthesized
    clone_audio: typing.Optional[str] = None  # Generated cloned audio as bytes

    # Fields for Type III (Deepfake Detection)
    detection_audio: typing.Optional[str] = None  # Audio data for detection
    detection_prediction: typing.Optional[int] = None  # Prediction value

    def deserialize(self) -> typing.Optional[str|bytes]:
        """
        Deserialize and return the appropriate output based on the synapse type.
        """
        if self.synapse_type == 'stt':
            return self.stt_transcription
        elif self.synapse_type == 'clone':
            return self.clone_audio.decode() if self.clone_audio else None
        elif self.synapse_type == 'detection':
            return str(self.detection_prediction)
        else:
            raise ValueError("Invalid synapse type")

    def is_valid_url(self) -> bool:
        """
        Check if the provided URL is valid for Type I (STT).
        """
        return bool(self.stt_link) and self.stt_link.startswith(('http://', 'https://'))

    def set_synapse_type(self, synapse_type) -> str:
        self.synapse_type = synapse_type
        
    def get_synapse_type(self) -> str:
        """
        Return the current synapse type.
        """
        return self.synapse_type