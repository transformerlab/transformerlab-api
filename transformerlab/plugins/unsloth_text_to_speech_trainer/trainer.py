from unsloth import FastModel
from abc import ABC, abstractmethod
from transformers import CsmForConditionalGeneration
import torch

class AudioTrainerBase(ABC):
    def __init__(self, model_name):
        self.model_name = model_name
        
    @abstractmethod
    def load_model(self):
        pass
    
    @abstractmethod
    def preprocess_dataset(self, example):
        pass
    
    @abstractmethod
    def setup_lora(self, model):
        pass
    
    @abstractmethod
    def train(self, dataset):
        pass

class CsmAudioTrainer(AudioTrainerBase):
    def load_model(self, max_seq_length):
        model, processor = FastModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=max_seq_length,
            dtype=None,  # Leave as None for auto-detection
            auto_model=CsmForConditionalGeneration,
            load_in_4bit=False,  # Keep this set to False because voice models are small, so we can maintain high quality results.
        )
        return model, processor
    
    def preprocess_dataset(self, example, speaker_key="source", processor=None, max_seq_length=256, max_audio_length=240001, sampling_rate=24000):
        conversation = [
            {
                "role": str(example[speaker_key]),
                "content": [
                    {"type": "text", "text": example["text"]},
                    {"type": "audio", "path": example["audio"]["array"]},
                ],
            }
        ]

        try:
            model_inputs = processor.apply_chat_template(
                conversation,
                tokenize=True,
                return_dict=True,
                output_labels=True,
                text_kwargs = {
                    "padding": "max_length",  # pad to the max_length
                    "max_length": max_seq_length, # this should be the max length of audio
                    "pad_to_multiple_of": 8, # Pad so length is a multiple of 8 (for efficiency)
                    "padding_side": "right",
                },
                audio_kwargs = {
                    "sampling_rate": sampling_rate,
                    "max_length": max_audio_length, # max input_values length of the whole dataset
                    "padding": "max_length",
                },
                common_kwargs = {"return_tensors": "pt"},
            )
        except Exception as e:
            print(f"Error processing example with text '{example['text'][:50]}...': {e}")
            return None

        required_keys = ["input_ids", "attention_mask", "labels", "input_values", "input_values_cutoffs"]
        processed_example = {}
        # print(model_inputs.keys())
        for key in required_keys:
            if key not in model_inputs:
                print(f"Warning: Required key '{key}' not found in processor output for example.")
                return None

            value = model_inputs[key][0]
            processed_example[key] = value


        # Final check (optional but good)
        if not all(isinstance(processed_example[key], torch.Tensor) for key in processed_example):
            print(f"Error: Not all required keys are tensors in final processed example. Keys: {list(processed_example.keys())}")
            return None

        return processed_example

class OrpheusAudioTrainer(AudioTrainerBase):
    # Your Orpheus-specific training logic  
    pass