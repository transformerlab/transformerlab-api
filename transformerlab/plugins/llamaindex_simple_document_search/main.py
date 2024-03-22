import argparse
import logging
import sys
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.callbacks import (
    CallbackManager,
    LlamaDebugHandler,
)
from llama_index.core import VectorStoreIndex
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
import os
from llama_index.core import SimpleDirectoryReader


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--documents_dir', default='', type=str, required=True)
parser.add_argument('--query', default='', type=str, required=True)
args, unknown = parser.parse_known_args()

# We must do exclude_hidden because ~.transformerlab has a . in its name
reader = SimpleDirectoryReader(
    input_dir=args.documents_dir, exclude_hidden=False)
documents = reader.load_data()
sys.stderr.write(f"Loaded {len(documents)} docs")

model_short_name = args.model_name.split("/")[-1]

# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
# )

# def messages_to_prompt(messages):
#     prompt = ""
#     for message in messages:
#         if message.role == 'system':
#             prompt += f"<|system|>\n{message.content}</s>\n"
#         elif message.role == 'user':
#             prompt += f"<|user|>\n{message.content}</s>\n"
#         elif message.role == 'assistant':
#             prompt += f"<|assistant|>\n{message.content}</s>\n"

#     # ensure we start with a system prompt, insert blank if needed
#     if not prompt.startswith("<|system|>\n"):
#         prompt = "<|system|>\n</s>\n" + prompt

#     # add final assistant prompt
#     prompt = prompt + "<|assistant|>\n"

#     return prompt

# Use the following to call HugggingFace Tranformers Directly.
# We will use OpenAILike instead.
# llm = HuggingFaceLLM(
#     model_name="HuggingFaceH4/zephyr-7b-alpha",
#     tokenizer_name="HuggingFaceH4/zephyr-7b-alpha",
#     query_wrapper_prompt=PromptTemplate(
#         "<|system|>\n</s>\n<|user|>\n{query_str}</s>\n<|assistant|>\n"),
#     context_window=3900,
#     max_new_tokens=256,
#     model_kwargs={"quantization_config": quantization_config},
#     # tokenizer_kwargs={},
#     generate_kwargs={"temperature": 0.7, "top_k": 50,
#                      "top_p": 0.95, "do_sample": True},
#     messages_to_prompt=messages_to_prompt,
#     device_map="auto",
# )


llm = OpenAILike(
    api_key="fake",
    api_type="fake",
    api_base="http://localhost:8000/v1",
    model=model_short_name,
    is_chat_model=True,
    timeout=40,
    # context_window=32000,
    tokenizer=model_short_name,
)

Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])

vector_index = VectorStoreIndex.from_documents(
    documents,  callback_manager=None)

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

query_engine = vector_index.as_query_engine(
    response_mode="compact")

response = query_engine.query(args.query)

print(response)
