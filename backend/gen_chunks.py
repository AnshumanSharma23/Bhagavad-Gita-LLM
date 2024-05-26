import os

# Function to read the text file
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# Function to split text into smaller chunks (e.g., by verses or paragraphs)
def split_text(text, chunk_size=1000):
    sentences = text.split('. ')
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)
        if current_length + sentence_length > chunk_size:
            chunks.append('. '.join(current_chunk) + '.')
            current_chunk = []
            current_length = 0
        current_chunk.append(sentence)
        current_length += sentence_length
    
    if current_chunk:
        chunks.append('. '.join(current_chunk) + '.')
    
    return chunks

# Function to save each chunk to a separate file
def save_chunks_to_files(chunks, directory):
    os.makedirs(directory, exist_ok=True)
    for i, chunk in enumerate(chunks):
        with open(os.path.join(directory, f'chunk_{i+1}.txt'), 'w', encoding='utf-8') as chunk_file:
            chunk_file.write(chunk)

# Path to the Bhagavad Gita text file
file_path = 'bhagavad_gita.txt'

# Directory to save the chunks
output_directory = 'data'

# Read the text file
text = read_text_file(file_path)

# Split text into chunks
chunks = split_text(text, chunk_size=1000)

# Save chunks to files
save_chunks_to_files(chunks, output_directory)

print(f'Successfully split the text into {len(chunks)} chunks and saved them in the "{output_directory}" directory.')
