from unsloth import FastModel
from abc import ABC, abstractmethod
from transformers import AutoProcessor, CsmForConditionalGeneration
from snac import SNAC
import torch
import librosa

class AudioModelBase(ABC):
    def __init__(self, model_name, device, context_length=2048):
        self.model_name = model_name
        self.device = device
        self.context_length = context_length

    @abstractmethod
    def tokenize(self, text, audio_path=None, sample_rate=24000):
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
        # [1]: https://github.com/canopyai/Orpheus-TTS/tree/main?tab=readme-ov-file#prompting
        self.generate_kwargs = {
            "max_new_tokens": 10240,
            "eos_token_id": 128258,
            "use_cache": True,
            "repetition_penalty": 1.1,
        }
        self.start_tokens = torch.tensor([[128259]], dtype=torch.int64)  # SOH (Start of Header)
        self.end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)  # EOT EOH
        self.final_tokens = torch.tensor([[128258, 128262]], dtype=torch.int64)
        self.pad_token = 128263


        # Define special tokens
        self.tokenizer_length = 128256
        self.start_of_text = 128000
        self.end_of_text = 128009
        self.start_of_speech = self.tokenizer_length + 1
        self.end_of_speech = self.tokenizer_length + 2
        self.start_of_human = self.tokenizer_length + 3
        self.end_of_human = self.tokenizer_length + 4
        self.start_of_ai = self.tokenizer_length + 5
        self.end_of_ai = self.tokenizer_length + 6
        self.audio_tokens_start = self.tokenizer_length + 10
        self.pad_token = self.tokenizer_length + 7
        self.ds_sample_rate = 24000

    def tokenize(self, text, audio_path=None, sample_rate=24000):
        """
        Tokenize text and optionally audio for voice cloning.
        
        Args:
            text: Text to convert to speech
            audio_path: Path to reference audio file for voice cloning
            sample_rate: Sample rate for audio processing
            
        Returns:
            Dictionary with input_ids and attention_mask for model generation
        """
        if audio_path:
            # Load and tokenize audio for voice cloning
            audio_array, _ = librosa.load(audio_path, sr=sample_rate)
            audio_tokens = self._tokenize_audio(audio_array)
            return self._create_voice_cloning_input(text, audio_tokens)
        else:
            # Standard text-to-speech without voice cloning
            return self.tokenizer(text, return_tensors="pt").to(self.device)

    def generate(self, inputs, **kwargs):
        """
        Generate audio tokens from inputs.
        
        Args:
            inputs: Tokenized inputs (dict with input_ids and attention_mask or legacy tensor)
            **kwargs: Additional generation arguments
            
        Returns:
            Generated token sequences
        """
        # Handle both new dict format and legacy tensor format
        if isinstance(inputs, dict):
            gen_args = {**inputs, **self.generate_kwargs, **kwargs}
        else:
            # Legacy format - convert to dict
            gen_args = {"input_ids": inputs, **self.generate_kwargs, **kwargs}
            
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
    
    def _tokenize_audio(self, waveform):
        """
        Convert audio waveform to tokens using SNAC encoder.
        
        Args:
            waveform: Audio waveform as numpy array
            
        Returns:
            List of audio tokens
        """
        waveform = torch.from_numpy(waveform).unsqueeze(0)
        waveform = waveform.to(dtype=torch.float32, device=self.device)
        waveform = waveform.unsqueeze(0)

        with torch.inference_mode():
            codes = self.snac_model.encode(waveform)

        # Convert codes to tokens with proper offsets
        all_codes = []
        for i in range(codes[0].shape[1]):
            all_codes.append(codes[0][0][i].item() + 128266)
            all_codes.append(codes[1][0][2*i].item() + 128266 + 4096)
            all_codes.append(codes[2][0][4*i].item() + 128266 + (2*4096))
            all_codes.append(codes[2][0][(4*i)+1].item() + 128266 + (3*4096))
            all_codes.append(codes[1][0][(2*i)+1].item() + 128266 + (4*4096))
            all_codes.append(codes[2][0][(4*i)+2].item() + 128266 + (5*4096))
            all_codes.append(codes[2][0][(4*i)+3].item() + 128266 + (6*4096))

        return all_codes
    
    def _create_voice_cloning_input(self, text, audio_tokens, voice_prompt="and_the_transcript_is"):
        """
        Create input for voice cloning by combining audio tokens and text.
        
        Args:
            text: Text to convert to speech
            audio_tokens: Tokenized audio for voice reference
            voice_prompt: Prompt describing the audio transcript
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        # Special tokens for voice cloning format
        

        # Tokenize voice prompt and target text
        voice_prompt_tokens = self.tokenizer(voice_prompt, return_tensors="pt").input_ids
        text_tokens = self.tokenizer(text, return_tensors="pt").input_ids
        
        # Create voice reference sequence: SOH + voice_prompt + EOT EOH + audio_tokens + final_tokens
        voice_reference = torch.cat([
            start_tokens,
            voice_prompt_tokens,
            end_tokens,
            torch.tensor([audio_tokens], dtype=torch.int64),
            final_tokens
        ], dim=1)
        
        # Create complete input: voice_reference + SOH + target_text + EOT
        input_ids = torch.cat([
            voice_reference,
            start_tokens,
            text_tokens,
            torch.tensor([[128009, 128260, 128261]], dtype=torch.int64)  # EOT
        ], dim=1)
        
        # Create attention mask (all ones for valid tokens)
        attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
        
        return {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device)
        }