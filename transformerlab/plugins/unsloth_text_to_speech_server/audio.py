from unsloth import FastModel
from abc import ABC, abstractmethod
from transformers import AutoProcessor, CsmForConditionalGeneration
from snac import SNAC
import torch

class AudioModelBase(ABC):
    def __init__(self, model_name, device, context_length=2048):
        self.model_name = model_name
        self.device = device
        self.context_length = context_length

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
    def __init__(self, model_name, device, context_length=2048):
        super().__init__(model_name, device, context_length)
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
            "max_new_tokens": 1024,
            "output_audio": True,
        }

    def tokenize(self, text):
        speaker_id = 0
        return self.processor(f"[{speaker_id}]{text}", add_special_tokens=True).to(self.device)

    def generate(self, inputs, **kwargs):
        gen_args = {**inputs, **self.generate_kwargs, **kwargs}
        return self.model.generate(**gen_args)

    def decode(self, generated, **kwargs):
        audio = generated[0].to(torch.float32).cpu().numpy()
        return audio

class OrpheusAudioModel(AudioModelBase):
    def __init__(self, model_name, device, context_length=2048):
        super().__init__(model_name, device, context_length)
        self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
        self.snac_model = self.snac_model.to(self.device)
        self.model, self.tokenizer = FastModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.context_length,
            dtype=None,
            load_in_4bit=False,
        )
        FastModel.for_inference(self.model)
        self.model = self.model.to(self.device)
        # Tips for prompting [1]:
        #  - Sampling parameters 'temperature' and 'top_p' work just like regular LLMs.
        #  - 'repetition_penalty' >= 1.1 is required for stable generations.
        #  - Increasing 'repetition_penalty' and/or 'temperature' makes the model speak faster.
        self.generate_kwargs = {
            "max_new_tokens": 10240,
            "eos_token_id": 128258,
            "use_cache": True,
            "repetition_penalty": 1.1,
        }

    def tokenize(self, text):

        return self.tokenizer(text, return_tensors="pt").to(self.device)

    def generate(self, inputs, **kwargs):
        gen_args = {**inputs, **self.generate_kwargs, **kwargs}
        return self.model.generate(**gen_args)

    def decode(self, generated_ids, **kwargs):

        #Post-process: remove unwanted tokens before decoding to audio
        token_to_find = 128257  # <start_of_speech>
        token_to_remove = 128258  # <end_of_speech>
        # Look for the special CODE_START_TOKEN_ID (128257 in this model) which marks the beginning of the audio token sequence.
        token_indices = (generated_ids == token_to_find).nonzero(as_tuple=True)
        if len(token_indices[1]) > 0:
            last_occurrence_idx = token_indices[1][-1].item()
            cropped_tensor = generated_ids[:, last_occurrence_idx+1:]
        else:
            cropped_tensor = generated_ids

        processed_rows = [row[row != token_to_remove] for row in cropped_tensor]

        # Convert generated tokens into codec codes
        code_lists = []
        for row in processed_rows:
            row_length = row.size(0)
            new_length = (row_length // 7) * 7
            trimmed_row = row[:new_length]
            # Offset them (subtract CODE_TOKEN_OFFSET which is 128266 here) to get the actual code values (0-4095 range)
            trimmed_row = [t - 128266 for t in trimmed_row]  # Adjust to codec IDs
            code_lists.append(trimmed_row)

        # Run decoding
        audio = [self.redistribute_codes(code_list) for code_list in code_lists]
        return audio[0].squeeze().to(torch.float32).cpu().detach().numpy()
    
    def redistribute_codes(self, code_list):
        """
        Decode to waveform using SNAC.
        
        The LLM outputs audio tokens, not playable sound. We need an encoder/decoder 
        to convert these tokens into an audio waveform. SNAC expects audio tokens 
        grouped in a specific structure (7 tokens per group across 3 layers in this setup).
        """
        layer_1, layer_2, layer_3 = [], [], []
        for i in range((len(code_list)+1)//7):
            layer_1.append(code_list[7*i])
            layer_2.append(code_list[7*i+1] - 4096)
            layer_3.append(code_list[7*i+2] - (2*4096))
            layer_3.append(code_list[7*i+3] - (3*4096))
            layer_2.append(code_list[7*i+4] - (4*4096))
            layer_3.append(code_list[7*i+5] - (5*4096))
            layer_3.append(code_list[7*i+6] - (6*4096))
        codes = [torch.tensor(layer_1, device=self.device).unsqueeze(0),
                torch.tensor(layer_2, device=self.device).unsqueeze(0),
                torch.tensor(layer_3, device=self.device).unsqueeze(0)]
        return self.snac_model.decode(codes)