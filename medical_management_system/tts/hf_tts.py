import struct
import wave
from typing import Iterable


class HuggingFaceTTS:
    """Generate speech audio with Hugging Face text-to-speech models."""

    def __init__(self, model_name: str = "facebook/mms-tts-eng") -> None:
        self.model_name = model_name
        self._pipeline = None

    def _get_pipeline(self):
        if self._pipeline is None:
            from transformers import pipeline

            self._pipeline = pipeline(task="text-to-speech", model=self.model_name)
        return self._pipeline

    @staticmethod
    def _to_pcm16(samples: Iterable[float]) -> bytes:
        pcm = bytearray()
        for sample in samples:
            clipped = max(-1.0, min(1.0, float(sample)))
            pcm.extend(struct.pack("<h", int(clipped * 32767)))
        return bytes(pcm)

    def text_to_speech(self, text: str, output_path: str) -> str:
        result = self._get_pipeline()(text)
        audio = result["audio"]
        sample_rate = int(result["sampling_rate"])
        if hasattr(audio, "tolist"):
            audio = audio.tolist()

        with wave.open(output_path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(self._to_pcm16(audio))
        return output_path

