# model_helper - common functions for using models from various sources

from transformerlab.models import ollamamodel
from transformerlab.models import huggingfacemodel

def list_model_sources():
    return [
        "huggingface",
        "ollama"
    ]


async def list_models_from_source(model_source: str, uninstalled_only: bool = True):
    """
    Wrapper to have a single funciton for getting lists of models.
    Should architect this more like plugins so we can dynamically 
    add and remove sources.
    """
    try:
        match model_source:
          case "ollama":
            return await ollamamodel.list_models(uninstalled_only)
          case "huggingface":
            return await huggingfacemodel.list_models(uninstalled_only)
    except Exception as e:
        print(e)
    return []


def model_architecture_is_supported(model_architecture: str):
    # Return true if the passed string is a supported model architecture
    supported_architectures = [
        "GGUF",
        "MLX",
        "LlamaForCausalLM",
        "T5ForConditionalGeneration",
        "FalconForCausalLM",
        "MistralForCausalLM",
        "MixtralForCausalLM",
        "GPTBigCodeForCausalLM",
        "GemmaForCausalLM",
        "CohereForCausalLM",
        "PhiForCausalLM",
        "Phi3ForCausalLM"

    ]
    return model_architecture in supported_architectures
