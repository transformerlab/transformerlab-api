from abc import ABC, abstractmethod
from unsloth import FastModel
from transformers import AutoProcessor, TrainingArguments, Trainer
import torch

class AudioTrainerBase(ABC):
    def __init__(self, model_name, config):
        self.model_name = model_name
        self.config = config
        
    @abstractmethod
    def load_model(self):
        pass
    
    @abstractmethod
    def preprocess_dataset(self, dataset):
        pass
    
    @abstractmethod
    def setup_lora(self, model):
        pass
    
    @abstractmethod
    def train(self, dataset):
        pass

class CsmAudioTrainer(AudioTrainerBase):
    # Your CSM-specific training logic
    pass

class OrpheusAudioTrainer(AudioTrainerBase):
    # Your Orpheus-specific training logic  
    pass