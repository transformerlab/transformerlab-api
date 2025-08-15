from abc import ABC, abstractmethod
from transformers import AutoProcessor, CsmForConditionalGeneration
from unsloth import FastModel
from snac import SNAC
import torch
import re

def convert_to_audio_snac(audio_ids, model):
    audio_ids = torch.tensor(audio_ids, dtype=torch.int32).reshape(-1, 7)
    codes_0 = audio_ids[:, 0].unsqueeze(0)
    codes_1 = torch.stack((audio_ids[:, 1], audio_ids[:, 4])).t().flatten().unsqueeze(0)
    codes_2 = (
        torch.stack((audio_ids[:, 2], audio_ids[:, 3], audio_ids[:, 5], audio_ids[:, 6]))
        .t()
        .flatten()
        .unsqueeze(0)
    )

    with torch.inference_mode():
        audio_hat = model.decode([codes_0, codes_1, codes_2])

    return audio_hat[0]

class AudioModelBase(ABC):
    def __init__(self, model_name, device, context_length=2048, snac_model=None):
        self.model_name = model_name
        self.device = device
        self.context_length = context_length
        self.snac_model = snac_model

    @abstractmethod
    def tokenize(self, text):
        pass

    @abstractmethod
    def generate(self, inputs, **kwargs):
        pass

    @abstractmethod
    def decode(self, generated, **kwargs):
        pass

class CsmAudioModel(AudioModelBase):
    def __init__(self, model_name, device, context_length=2048, snac_model=None):
        super().__init__(model_name, device, context_length, snac_model)
        self.processor = AutoProcessor.from_pretrained(self.model_name)
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
            "max_new_tokens": 1200,
            "output_audio": True,
        }

    def tokenize(self, text):
        speaker_id = 0
        return self.processor(f"[{speaker_id}]{text}", add_special_tokens=True).to(self.device)

    def generate(self, inputs, **kwargs):
        # Merge self.generate_kwargs with any overrides in kwargs
        gen_args = {**inputs, **self.generate_kwargs, **kwargs}
        return self.model.generate(**gen_args)

    def decode(self, generated, **kwargs):
        audio = generated[0].to(torch.float32).cpu().numpy()
        return audio

class OrpheusAudioModel(AudioModelBase):
    def __init__(self, model_name, device, context_length=2048, snac_model=None):
        super().__init__(model_name, device, context_length, snac_model)
        self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
        self.snac_model = self.snac_model.to(self.device)
        # Assume self.model and self.tokenizer are set up elsewhere
        self.generate_kwargs = {
            "max_new_tokens": 10240,
            "eos_token_id": 128258,
            "use_cache": True,
        }

    def tokenize(self, text):
        return self.tokenizer(text, return_tensors="pt").to(self.device)

    def generate(self, inputs, **kwargs):
        gen_args = {**inputs, **self.generate_kwargs, **kwargs}
        return self.model.generate(**gen_args)

    def decode(self, generated, **kwargs):
        AUDIO_TOKENS_REGEX = re.compile(r"<custom_token_(\d+)>")
        audio_ids = [
            int(token) - 10 - ((index % 7) * 4096)
            for index, token in enumerate(AUDIO_TOKENS_REGEX.findall(generated))
        ]
        audio = convert_to_audio_snac(audio_ids, self.snac_model)
        return audio