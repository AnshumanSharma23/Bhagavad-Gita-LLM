import os
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
import openai
import numpy as np
import sqlite3
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = ""
openai.api_key = os.getenv("OPENAI_API_KEY")

# Set the global default embedding model
Settings.embed_model = OpenAIEmbedding()

# Rebuild storage context
persist_dir = "./persisted_index"
storage_context = StorageContext.from_defaults(persist_dir=persist_dir)

# Load index
index = load_index_from_storage(storage_context)

# Create query engine
query_engine = index.as_query_engine()

# Function to generate embeddings using a consistent model
def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    response = openai.embeddings.create(input=[text], model=model)
    embedding = np.array(response.data[0].embedding)
    print(f"Generated embedding shape: {embedding.shape}")  # Debug information
    return embedding
    

# Converting the np array to blob
def np_to_blob(array):
    print(f"Converting np array to blob, shape: {array.shape}")  # Debug information
    return array.astype(np.float32).tobytes()

def blob_to_np(blob):
    array = np.frombuffer(blob, dtype=np.float32)
    print(f"Converted blob to np array, shape: {array.shape}")  # Debug information
    return array

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    user_query = data.get('query')
    
    # Generate embedding for the current query
    query_embedding = get_embedding(user_query)
    
    # Connect to the database
    conn = sqlite3.connect('feedback_of_users.db')
    c = conn.cursor()
    
    # Query the Llama index
    response = query_engine.query(user_query)
    formatted_response = f"User Query: {user_query}\n\nIndexed Response from Llama: {response}\n\n"
    
    # Retrieve all past queries and their embeddings
    c.execute('SELECT query, response, liked, embedding FROM feedback_of_users')
    all_records = c.fetchall()
    
    # Calculate similarity scores
    liked_similarities = []
    disliked_similarities = []
    
    for record in all_records:
        past_query, past_response, liked, embedding_blob = record
        past_embedding = blob_to_np(embedding_blob)
        similarity = cosine_similarity(query_embedding, past_embedding)
        
        if liked:
            liked_similarities.append((similarity, past_query, past_response, liked))
        else:
            disliked_similarities.append((similarity, past_query, past_response, liked))
    
    # Sort by similarity score in descending order
    liked_similarities.sort(reverse=True, key=lambda x: x[0])
    disliked_similarities.sort(reverse=True, key=lambda x: x[0])
    
    # Retrieve top 2 liked and top 1 disliked similar queries
    top_liked = liked_similarities[:2]
    top_disliked = disliked_similarities[:1]
    
    # Construct the context
    context = f"""
    User Query: {user_query}
    Similar Liked Queries and Responses:
    """
    
    for idx, sim in enumerate(top_liked):
        context += f"\n{idx+1}. Query: {sim[1]}\nResponse: {sim[2]}\nUser Feedback: Liked\n"
    
    context += "\nSimilar Disliked Query and Response:\n"
    for idx, sim in enumerate(top_disliked):
        context += f"\n{idx+1}. Query: {sim[1]}\nResponse: {sim[2]}\nUser Feedback: Disliked\n"
    
    # Include the initial context
    initial_context = """
    You are an AI assistant trained to provide answers based on the Bhagavad Gita. 
    When a user asks a question, you should respond with relevant passages or explanations from the Bhagavad Gita text.
    Try to answer as if Lord Krishna is trying to answer.
    """
    
    # Combine initial context with constructed context and formatted response
    full_context = f"{initial_context}\n{context}\n{formatted_response}"
    print(full_context)
    
    # Query the LLM with the full context
    openai.api_key = os.getenv("OPENAI_API_KEY")
    chat_response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": full_context}]
    )
    generated_response = chat_response.choices[0].message.content
    
    conn.close()
    return jsonify({"response": generated_response})

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.json
    user_query = data.get('query')
    response = data.get('response')
    liked = data.get('liked')
    query_embedding = get_embedding(user_query)
    query_embedding_blob = np_to_blob(query_embedding)
    try:
        conn = sqlite3.connect('feedback_of_users.db')
        c = conn.cursor()
        c.execute('''
        CREATE TABLE IF NOT EXISTS feedback_of_users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT,
            response TEXT,
            liked BOOLEAN,
            embedding BLOB
        )
        ''')
        c.execute('''
        INSERT INTO feedback_of_users (query, response, liked, embedding)
        VALUES (?, ?, ?, ?)
        ''', (user_query, response, liked, query_embedding_blob))
        conn.commit()
        conn.close()
        return jsonify({"message": "Feedback submitted successfully"}), 200
    except Exception as e:
        return jsonify({"message": f"Error inserting into database: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True)


































# import os
# from llama_index.core import StorageContext, load_index_from_storage
# from llama_index.embeddings.openai import OpenAIEmbedding
# from llama_index.core import Settings
# import openai
# import numpy as np
# import sqlite3
# from flask import Flask, request, jsonify
# from flask_cors import CORS


# app = Flask(__name__)


# # Set OpenAI API key
# os.environ["OPENAI_API_KEY"] = "sk-proj-pcZ2iOvefaLNdQxvTz0ZT3BlbkFJlhMnAUOShF9nYCjNVFM1"
# openai.api_key = os.getenv("OPENAI_API_KEY")

# # Set the global default embedding model
# Settings.embed_model = OpenAIEmbedding()

# # Rebuild storage context
# persist_dir = "./persisted_index"
# storage_context = StorageContext.from_defaults(persist_dir=persist_dir)

# # Load index
# index = load_index_from_storage(storage_context)

# # Create query engine
# query_engine = index.as_query_engine()

# # Creating a function to generate embeddings

# def get_embedding(text, model="text-embedding-3-small"):
#     text = text.replace("\n", " ")
#     response = openai.embeddings.create(input=[text], model=model)
#     return np.array(response.data[0].embedding)


# # converting the np array to blob
# def np_to_blob(array):
#     return array.tobytes()

# def blob_to_np(blob):
#     return np.frombuffer(blob, dtype=np.float32)

# def cosine_similarity(a, b):
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))








# @app.route('/query', methods=['POST'])
# def query():
#     data = request.json
#     user_query = data.get('query')
    
#     # Generate embedding for the current query
#     query_embedding = get_embedding(user_query)
    
#     # Connect to the database
#     conn = sqlite3.connect('feedback_of_users.db')
#     c = conn.cursor()
#     response = query_engine.query(user_query)
#     formatted_response = f"User Query: {user_query}\n\nIndexed Response that i got from the vector db/documents i have from the bhagwad gita are : {response}\n\n"
    
#     # Retrieve all past queries and their embeddings
#     c.execute('SELECT query, response, liked, embedding FROM feedback_of_users')
#     all_records = c.fetchall()
    
#     # Calculate similarity scores
#     liked_similarities = []
#     disliked_similarities = []
    
#     for record in all_records:
#         past_query, past_response, liked, embedding_blob = record
#         past_embedding = blob_to_np(embedding_blob)
#         similarity = cosine_similarity(query_embedding, past_embedding)
        
#         if liked:
#             liked_similarities.append((similarity, past_query, past_response, liked))
#         else:
#             disliked_similarities.append((similarity, past_query, past_response, liked))
    
#     # Sort by similarity score in descending order
#     liked_similarities.sort(reverse=True, key=lambda x: x[0])
#     disliked_similarities.sort(reverse=True, key=lambda x: x[0])
    
#     # Retrieve top 2 liked and top 1 disliked similar queries
#     top_liked = liked_similarities[:2]
#     top_disliked = disliked_similarities[:1]
    
#     # Construct the context
#     context = f"""
#     User Query: {user_query}
#     Similar Liked Queries and Responses:
#     """
    
#     for idx, sim in enumerate(top_liked):
#         context += f"\n{idx+1}. Query: {sim[1]}\nResponse: {sim[2]}\nUser Feedback: Liked\n"
    
#     context += "\nSimilar Disliked Query and Response:\n"
#     for idx, sim in enumerate(top_disliked):
#         context += f"\n{idx+1}. Query: {sim[1]}\nResponse: {sim[2]}\nUser Feedback: Disliked\n"
    
#     # Include the initial context
#     initial_context = """
#     You are an AI assistant trained to provide answers based on the Bhagavad Gita. 
#     When a user asks a question, you should respond with relevant passages or explanations from the Bhagavad Gita text.
#     Try to answer as if Lord Krishna is trying to answer.
#     """
    
#     # Combine initial context with constructed context
#     full_context = f"{initial_context}\n{context}"
    
#     # Query the LLM with the full context
#     openai.api_key = os.getenv("OPENAI_API_KEY")
#     chat_response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=[{"role": "system", "content": full_context}
#                   ,{"role": "user", "content": formatted_response}]
#     )
#     generated_response = chat_response.choices[0].message.content
    
#     conn.close()
#     return jsonify({"response": generated_response})








# @app.route('/feedback', methods=['POST'])
# def feedback():
#     data = request.json
#     user_query = data.get('query')
#     response = data.get('response')
#     liked = data.get('liked')
#     query_embedding = get_embedding(user_query)
#     query_embedding_blob = np_to_blob(query_embedding)
#     try:
#         conn = sqlite3.connect('feedback_of_users.db')
#         c = conn.cursor()
#         c.execute('''
#         CREATE TABLE IF NOT EXISTS feedback_of_users (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             query TEXT,
#             response TEXT,
#             liked BOOLEAN,
#             embedding BLOB
#         )
#         ''')
#         c.execute('''
#         INSERT INTO feedback_of_users (query, response, liked, embedding)
#         VALUES (?, ?, ?, ?)
#         ''', (user_query, response, liked, query_embedding_blob))
#         conn.commit()
#         conn.close()
#         return jsonify({"message": "Feedback submitted successfully"}), 200
#     except Exception as e:
#         return jsonify({"message": f"Error inserting into database: {e}"}), 500

# if __name__ == '__main__':
#     app.run(debug=True)



    





# @app.route('/query', methods=['POST'])
# def query():
#     data = request.json
#     user_query = data.get('query')
#     context = """
#     You are an AI assistant trained to provide answers based on the Bhagavad Gita. 
#     When a user asks a question, you should respond with relevant passages or explanations from the Bhagavad Gita text.
#     Try to answer as if Lord Krishna is trying to answer. 
#     """
#     response = query_engine.query(user_query)
#     formatted_response = f"User Query: {user_query}\n\nIndexed Response that i got from the vector db: {response}\n\n"
#     openai.api_key = os.getenv("OPENAI_API_KEY")
#     chat_response = openai.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=[{"role": "system", "content": context},
#                   {"role": "user", "content": formatted_response}]
#     )   
#     generated_response = chat_response.choices[0].message.content
#     return jsonify({"response": generated_response})




# # # Provide additional context to OpenAI LLM
# # context = """
# # You are an AI assistant trained to provide answers based on the Bhagavad Gita. 
# # When a user asks a question, you should respond with relevant passages or explanations from the Bhagavad Gita text.
# # Try to answer as if Lord Krishna is trying to answer. 
# # """

# # # If user submits a query
# # if st.button("Submit"):
# #     if user_query:
# #         # Query the index
# #         response = query_engine.query(user_query)
# #         formatted_response = f"User Query: {user_query}\n\nIndexed Response that i got from the vector db: {response}\n\n{context}"
# #         # Format the response with additional context
# #         openai.api_key = os.getenv("OPENAI_API_KEY")
# #         chat_response = openai.chat.completions.create(
# #         model="gpt-3.5-turbo",
# #         messages=[{"role": "system", "content": context},
# #                   {"role": "user", "content": formatted_response}]
# #     )   
        
# #         generated_response = chat_response.choices[0].message.content
# #         # print(response)
# #         # formatted_response=response.choices[0].message.content.strip()
# #         # generated_response = response.choices[0].message.content.strip()
# #         st.write(f"Response: {generated_response}")
# #         # feedback = st.radio("Do you like this response?", ("Like", "Dislike"))
# #         st.write("Do you like this response?")
        
# #         feedback = st.radio("Do you like this response?", ("Like", "Dislike"))

# #         if feedback is not None:
# #             liked = feedback == "Like"
            
# #             query_embedding = get_embedding(user_query)
# #             query_embedding_blob = np_to_blob(query_embedding)

# #             st.write(f"user_query: {user_query}")
# #             st.write(f"generated_response: {generated_response}")
# #             st.write(f"liked: {liked}")
# #             st.write(f"query_embedding_blob (length): {len(query_embedding_blob)}")

# #             try:
                    
# #                 conn = sqlite3.connect('feedback_of_users.db')
# #                 c = conn.cursor()
# #                 c.execute('''
# #                 CREATE TABLE IF NOT EXISTS feedback_of_users (
# #                     id INTEGER PRIMARY KEY AUTOINCREMENT,
# #                     query TEXT,
# #                     response TEXT,
# #                     liked BOOLEAN,
# #                     embedding BLOB
# #                 )
# #                 ''')
# #                 st.write("Inserting data into database")
# #                 c.execute('''
# #                 INSERT INTO feedback_of_users (query, response, liked, embedding)
# #                 VALUES (?, ?, ?, ?)
# #                 ''', (user_query, generated_response, liked, query_embedding_blob))
# #                 conn.commit()
# #                 st.write("Data committed to database")
# #                 conn.close()
# #                 st.write("Feedback submitted, Thank you :) ")
# #                 print("feedback added to the db")

# #             except Exception as e:
# #                 st.write(f"Error inserting into database: {e}")
# #     else:
# #         st.write("Please enter a query.")

# from flask import Flask, request, jsonify

# app = Flask(__name__)

# @app.route('/members', methods=['GET'])
# def members():
#     return jsonify({"members": ["John Doe", "Jane Doe", "Jim Doe"]})

# @app.route('/query', methods=['POST'])
# def query():
#     try:
#         data = request.json
#         user_query = data.get('query')
#         print(f"Received query: {user_query}")
#         response = {"response": "Your name is John Doe."}
#         return jsonify(response)
#     except Exception as e:
#         print(f"Error during query: {e}")
#         return jsonify({"error": str(e)}), 500

# @app.route('/feedback', methods=['POST'])
# def feedback():
#     try:
#         data = request.json
#         user_query = data.get('query')
#         response = data.get('response')
#         liked = data.get('liked')
#         print(f"Received feedback: Query={user_query}, Response={response}, Liked={liked}")
#         # Here you can add logic to store feedback in the database
#         return jsonify({"message": "Feedback submitted successfully"}), 200
#     except Exception as e:
#         print(f"Error during feedback: {e}")
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)

