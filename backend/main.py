import os
import streamlit as st
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = ""

# Set the global default embedding model
Settings.embed_model = OpenAIEmbedding()

# Rebuild storage context
persist_dir = "./persisted_index"
storage_context = StorageContext.from_defaults(persist_dir=persist_dir)

# Load index
index = load_index_from_storage(storage_context)

# Create query engine
query_engine = index.as_query_engine()

# Streamlit application
st.title("Bhagavad Gita Query Application")

st.write("""
This application allows you to query the Bhagavad Gita text using OpenAI's language model.
Enter your query below and get responses based on the indexed text.
""")

# Get user query
user_query = st.text_input("Enter your query:")

# Provide additional context to OpenAI LLM
context = """
You are an AI assistant trained to provide answers based on the Bhagavad Gita. 
When a user asks a question, you should respond with relevant passages or explanations from the Bhagavad Gita text.
Try to answer as if Lord Krishna is trying to answer.
"""

# If user submits a query
if st.button("Submit"):
    if user_query:
        # Query the index
        response = query_engine.query(user_query)
        
        # Format the response with additional context
        openai.api_key = os.getenv("OPENAI_API_KEY")
        formatted_response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"{context}\n\nUser Query: {user_query}\n\nResponse: {response}",
            max_tokens=150
        ).choices[0].text.strip()
        
        # Display the response
        st.write(f"Response: {formatted_response}")
    else:
        st.write("Please enter a query.")

if __name__ == "__main__":
    st.run()
