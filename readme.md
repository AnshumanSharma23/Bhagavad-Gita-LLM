

# Bhagavad Gita Query Application

The Bhagavad Gita Query Application is an intelligent system designed to provide insightful responses to user queries based on the Bhagavad Gita, leveraging advanced natural language processing (NLP) and machine learning techniques. The application also incorporates user feedback to continually improve and align responses with user preferences.

## Features

- **Query Processing**: Handles user queries and retrieves relevant passages from the Bhagavad Gita using the LlamaIndex library.
- **Response Generation**: Utilizes OpenAI's GPT-3.5-turbo model to generate contextually appropriate responses.
- **Feedback Mechanism**: Allows users to provide feedback on the responses to refine future interactions.
- **Embedding Management**: Employs OpenAI's text-embedding-ada-002 model to generate and manage text embeddings.

## Prerequisites

- Python 3.7 or higher
- Node.js and npm (for the frontend)
- An OpenAI API key

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com//bhagavad-gita-query-app.git
   cd bhagavad-gita-query-app
   ```

2. **Set up the backend**:

   - Navigate to the backend directory:

     ```bash
     cd backend
     ```

   - Create a virtual environment and activate it:

     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

   - Install the required packages:

     ```bash
     pip install -r requirements.txt
     ```

   - Add your OpenAI API key in `app.py` ,`gen_embed.py` :

     ```python
     os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
     ```

3. **Set up the frontend**:

   - Navigate to the frontend directory:

     ```bash
     cd ../frontend
     ```

   - Install the required packages:

     ```bash
     npm install
     ```

## Usage

1. **Run the backend**:

   ```bash
   cd backend
   flask run
   ```

2. **Run the frontend**:

   ```bash
   cd frontend
   npm start
   ```

3. **Access the application**:

   Open your browser and navigate to `http://localhost:3000`.

## File Structure

```plaintext
bhagavad-gita-query-app/
├── backend/
│   ├── app.py
│   ├── database.py
│   ├── gen_chunks.py
│   ├── gen_embed.py
│   ├── persisted_index/
│   └── feedback_of_users.db
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── App.css
│   │   ├── App.js
│   │   ├── index.css
│   │   └── index.js
│   ├── package.json
│   └── ... (other React files)
```

## Detailed Explanation

### Backend

- **gen_chunks.py**: Reads the Bhagavad Gita text, splits it into smaller chunks, and saves them as separate files.

  ```python
  # Function to split text into smaller chunks (e.g., by verses or paragraphs)
  def split_text(text, chunk_size=1000):
      ...
  ```

- **gen_embed.py**: Generates embeddings for the text chunks using OpenAI's embedding model and creates a vector store index.

  ```python
  from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
  ```

- **database.py**: Sets up the SQLite database to store queries, responses, and feedback with embeddings.

  ```python
  import sqlite3
  ```

- **app.py**: The main application file that handles query processing, response generation, and feedback integration.

  ```python
  from flask import Flask, request, jsonify
  from flask_cors import CORS
  ```

### Frontend

- **App.js**: The main React component that handles user input, displays responses, and manages feedback.

  ```jsx
  import React, { useState } from 'react';
  import axios from 'axios';
  ```

- **App.css**: Contains the styles for the application, making it visually appealing and user-friendly.

  ```css
  .app-container {
    ...
  }
  ```

## Contributing

If you'd like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -am 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Create a new Pull Request.


