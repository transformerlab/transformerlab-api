import argparse
import json
import sys
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.callbacks import (
    CallbackManager,
    LlamaDebugHandler,
    CBEventType
)
from llama_index.core import VectorStoreIndex
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
import os
from llama_index.core import SimpleDirectoryReader

# Redirect all output to a buffer that we control:


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--documents_dir', default='', type=str, required=True)
parser.add_argument('--query', default='', type=str, required=True)
parser.add_argument('--settings', default='', type=str, required=False)

args, unknown = parser.parse_known_args()

# SETTINGS
number_of_search_results = 2

Settings.context_window = 4096
Settings.num_output = 256

Settings.chunk_size = 512
Settings.chunk_overlap = 50

response_mode = "compact"

temperature = 0.7
# END SETTINGS

if args.settings:
    settings_param = json.loads(args.settings)
    if ("number_of_search_results" in settings_param):
        number_of_search_results = int(
            settings_param["number_of_search_results"])
    if ("context_window" in settings_param):
        Settings.context_window = int(settings_param["context_window"])
    if ("num_output" in settings_param):
        Settings.num_output = int(settings_param["num_output"])
    if ("chunk_size" in settings_param):
        Settings.chunk_size = int(settings_param["chunk_size"])
    if ("chunk_overlap" in settings_param):
        Settings.chunk_overlap = int(settings_param["chunk_overlap"])
    if ("response_mode" in settings_param):
        response_mode = settings_param["response_mode"]
    if ("temperature" in settings_param):
        temperature = float(settings_param["temperature"])


llama_debug = LlamaDebugHandler(print_trace_on_end=False)
callback_manager = CallbackManager([llama_debug])

# We must do exclude_hidden because ~.transformerlab has a . in its name
reader = SimpleDirectoryReader(
    input_dir=args.documents_dir, exclude_hidden=False)
documents = reader.load_data()
sys.stderr.write(f"Loaded {len(documents)} docs")

model_short_name = args.model_name.split("/")[-1]


llm = OpenAILike(
    api_key="fake",
    api_type="fake",
    api_base="http://localhost:8000/v1",
    model=model_short_name,
    is_chat_model=True,
    timeout=40,
    # context_window=32000,
    tokenizer=model_short_name,
    temperature=temperature,
)

Settings.llm = llm

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)
Settings.callback_manager = callback_manager

vector_index = VectorStoreIndex.from_documents(
    documents, required_exts=[".txt", ".pdf", ".docx", ".csv", ".epub", ".ipynb", ".mbox", ".md", ".ppt", ".pptm", ".pptx"])

query_engine = vector_index.as_query_engine(
    response_mode=response_mode, similarity_top_k=number_of_search_results)

rag_response = query_engine.query(args.query)

script_response = {}
script_response["response"] = rag_response.__str__()

events_to_track = [
    CBEventType.RETRIEVE, CBEventType.LLM, CBEventType.QUERY, CBEventType.TEMPLATING]

# events = llama_debug.get_events()
event_pairs = llama_debug.get_event_pairs()

for event_pair in event_pairs:
    # print(event_pair[0].event_type)
    # if event_pair[0].event_type in events_to_track:
    #     print(event_pair[0].payload)

    if event_pair[0].event_type == CBEventType.RETRIEVE:
        # print("\nRETRIEVE:")
        script_response["context"] = []
        nodes = event_pair[1].payload.get("nodes")
        for node in nodes:
            # print(node)
            script_response["context"].append(node.__str__())

    if event_pair[0].event_type == CBEventType.TEMPLATING:
        # print("\nTEMPLATE:")
        # print(event_pair[0].payload.keys())
        # print(event_pair[0].payload["template"])
        # print(event_pair[0].payload["template_vars"])
        # print(event_pair[0].payload["system_prompt"])
        script_response["template"] = event_pair[0].payload["template"].__str__()

print(json.dumps(script_response))
