import os
import torch
from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import AutoTokenizer
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# load the documents
documents = SimpleDirectoryReader(input_files=["Simple_RAG/Data/text.txt"]).load_data()

# Bge small model is used as embedder
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# huggingface token to access llama3
HF_TOKEN = os.environ['HF_TOKEN']


tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    token=HF_TOKEN,
)

stopping_ids = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
]

# llama3 is used as llm after retrieving chunks
llm = HuggingFaceLLM(
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    model_kwargs={
        "token": HF_TOKEN,
        "torch_dtype": torch.bfloat16, 
    },
    generate_kwargs={
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.9,
    },
    tokenizer_name="meta-llama/Meta-Llama-3-8B-Instruct",
    tokenizer_kwargs={"token": HF_TOKEN},
    stopping_ids=stopping_ids,
)


Settings.embed_model = embed_model
Settings.llm = llm

# store the documents
index = VectorStoreIndex.from_documents(documents,)

# establish the query engine
query_engine = index.as_query_engine(similarity_top_k=3)