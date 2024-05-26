import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = ""

# Set the global default embedding model
Settings.embed_model = OpenAIEmbedding()

# Load documents from the directory
documents = SimpleDirectoryReader("./data").load_data()

# Create the index from the documents
index = VectorStoreIndex.from_documents(documents)

# Persist the index to disk
persist_dir = "./persisted_index"
index.storage_context.persist(persist_dir=persist_dir)

# Example query
query_engine = index.as_query_engine()
response = query_engine.query("What is the meaning of life?")
print(response)
