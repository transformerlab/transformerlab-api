from abc import ABC, abstractmethod
import os
from transformers import AutoProcessor, CsmForConditionalGeneration
import torch
import librosa
import numpy as np


class AudioModelBase(ABC):
    def __init__(self, model_name, device, context_length=2048):
        self.model_name = model_name
        self.device = device
        self.context_length = context_length

    @abstractmethod
    def tokenize(self, text, audio_path=None, sample_rate=24000, voice=None):
        pass

    @abstractmethod
    def generate(self, inputs, **kwargs):
        pass

    @abstractmethod
    def decode(self, generated, **kwargs):
        pass


class CsmAudioModel(AudioModelBase):
    def __init__(self, model_name, device, processor_name, context_length=2048):
        super().__init__(model_name, device, context_length)
        from unsloth import FastModel

        self.processor = AutoProcessor.from_pretrained(processor_name)

        self.model, self.tokenizer = FastModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.context_length,
            dtype=None,
            auto_model=CsmForConditionalGeneration,
            load_in_4bit=False,
        )
        FastModel.for_inference(self.model)
        self.model = self.model.to(self.device)
        self.generate_kwargs = {
            "max_new_tokens": 1024,
            "output_audio": True,
        }

    def tokenize(self, text, audio_path=None, sample_rate=24000, voice=None):
        """
        Tokenize text and optionally audio for voice cloning.

        Args:
            text (str): Text to convert to speech
            audio_path (str, optional): Path to reference audio file for voice cloning
            sample_rate (int, optional): Sample rate for audio processing

        Returns:
            dict: Tokenized inputs ready for generation
        """
        speaker_id = 0

        if audio_path:
            # Load reference audio for voice cloning
            audio_array, _ = librosa.load(audio_path, sr=sample_rate)

            # Create conversation with reference audio and target text
            conversation = [
                {
                    "role": f"{speaker_id}",
                    "content": [
                        {"type": "text", "text": "This is how I sound."},
                        {"type": "audio", "path": audio_array},
                    ],
                },
                {
                    "role": f"{speaker_id}",
                    "content": [{"type": "text", "text": text}],
                },
            ]

            # Use processor's chat template for voice cloning
            inputs = self.processor.apply_chat_template(
                conversation,
                tokenize=True,
                return_dict=True,
            )
            return inputs.to(self.device)
        else:
            # Standard text-to-speech without voice cloning
            return self.processor(f"[{speaker_id}]{text}", add_special_tokens=True).to(self.device)

    def generate(self, inputs, **kwargs):
        gen_args = {**inputs, **self.generate_kwargs, **kwargs}
        return self.model.generate(**gen_args)

    def decode(self, generated, **kwargs):
        audio = generated[0].to(torch.float32).cpu().numpy()
        return audio


class OrpheusAudioModel(AudioModelBase):
    """
    Orpheus TTS wrapper that uses the official orpheus_tts package for inference.
    Generates streaming 24kHz mono int16 PCM and converts to numpy audio.
    """

    def __init__(self, model_name, device, context_length=2048):
        super().__init__(model_name, device, context_length)
        from orpheus_tts import OrpheusModel

        # Initialize official Orpheus model
        # max_model_len maps to context_length; device selection is handled by the package
        # Forward vLLM engine kwargs via **engine_kwargs supported by OrpheusModel
        # Defaults, with env var overrides for flexibility
        self.model = OrpheusModel(
            self.model_name,
            max_model_len=72000,
            gpu_memory_utilization=0.75,
        )
        # self.model.engine_kwargs = {
        #     "max_model_len": self.context_length,
        #     "gpu_memory_utilization": 0.95,
        # }
        # self.model._setup_engine()

    def tokenize(self, text, audio_path=None, sample_rate=24000, voice=None):
        # Package handles prompt formatting internally. We simply pass through.
        return {
            "prompt": text,
            "voice": voice,
            "sample_rate": sample_rate,
        }

    def generate(self, inputs, **kwargs):
        prompt = inputs.get("prompt", "hello 123")
        voice = inputs.get("voice", "amelie")
        temperature = kwargs.get("temperature", 1.0)
        top_p = kwargs.get("top_p", 1.0)

        stream = self.model.generate_speech(
            prompt=prompt,
            voice=voice,
            temperature=temperature,
            top_p=top_p,
        )

        # Collect streaming PCM bytes
        pcm_bytes = bytearray()
        for chunk in stream:
            pcm_bytes.extend(chunk)
        return bytes(pcm_bytes), chunk

    def decode(self, generated_bytes, **kwargs):
        # Convert raw int16 PCM bytes to float32 numpy array in [-1, 1]
        int16_audio = np.frombuffer(generated_bytes, dtype=np.int16)
        audio = (int16_audio.astype(np.float32) / 32768.0).copy()
        return audio
