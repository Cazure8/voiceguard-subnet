import os
from voiceguard.protocol import VoiceGuardSynapse
from voiceguard.utils.misc import is_youtube
from voiceguard.utils.helper import download_youtube_segment, transcribe_with_whisper

def speech_to_text(self, synapse: VoiceGuardSynapse) -> str:
    stt_link = synapse.stt_link
    segment = synapse.stt_segment

    if is_youtube(stt_link):
        try:
            output_filepath = download_youtube_segment(stt_link, segment)
            
            if not os.path.exists(output_filepath):
                print("Output file does not exist. Returning empty transcription.")
                start, _ = segment
                return format_transcription(start, "")
            
            transcription = transcribe_with_whisper(output_filepath)

            print("---miner transcript--")
            print(transcription)
            print("---------------------")

            start, _ = segment
            return format_transcription(start, transcription)
        
        except Exception as e:
            print(f"Failed during model loading or transcription: {e}")
            return ""

def format_transcription(segment_start, transcription):
    formatted_transcription = f"{segment_start}$$_{transcription}"
    return formatted_transcription