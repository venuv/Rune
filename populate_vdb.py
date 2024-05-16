# Import necessary modules and classes for DSPy and environment variable management
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate import answer_exact_match, answer_passage_match
from dspy import Example
import os
import argparse
import dspy
import pkg_resources
from dspy import Signature, InputField, OutputField, Module, Predict, Prediction
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from dotenv import load_dotenv

load_dotenv()


# Import LLaMA indexing utility for parsing documents
from llama_index.core import SimpleDirectoryReader

# Securely set the OpenAI API key as an environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")
llamaparse_api_key = os.getenv("LLAMAPARSE_API_KEY")

# Error handling for missing keys (optional)
if not openai_api_key or not llamaparse_api_key:
    raise ValueError("Missing required environment variables: OPENAI_API_KEY, LLAMAPARSE_API_KEY")

# Configure language models (LM) with specific parameters
# gpt-3.5-turbo: A fast version of GPT-3.5 for efficient text generation
turbo = dspy.OpenAI(model='gpt-3.5-turbo')
# gpt-4-1106-preview: A preview version of GPT-4 with a token limit for generating text
gpt4T = dspy.OpenAI(model='gpt-4-1106-preview', max_tokens=1000, model_type='chat')

# Set the default language model for DSPy settings
dspy.settings.configure(lm=turbo)

# Define command line arguments
parser = argparse.ArgumentParser(description="DSPy Teleprompt with secure API keys and directory argument")
parser.add_argument("directory", type=str, help="Path to the directory containing documents")

args = parser.parse_args()


# Initialize a parser for extracting text from files using the LLaMA model
# The parser is configured to use a specific API key and to output results in text format for English language
parser = LlamaParse(
    api_key=llamaparse_api_key,
    result_type="text",
    language="en",
    verbose=True  # Note: Corrected the typo from 'varbose' to 'verbose'
)

# Define a dictionary to map file extensions to their respective parser
# Currently, only PDF files are mapped to the LLaMA parser
file_extractor = {".pdf": parser}

# Initialize a directory reader to process files in a specified directory using the above mapping
reader = SimpleDirectoryReader(args.directory, file_extractor=file_extractor)


# Load data from files in the specified directory and parse them into documents
documents = reader.load_data()

# Output to console to indicate documents have been created successfully
print("Documents created")

# Initialize a vector index from the processed documents
# This index will allow for efficient similarity searches among the documents
index = VectorStoreIndex.from_documents(documents)

# Set an identifier for the vector index
index.set_index_id("vector_index")

# Persist the index and its data to the specified storage path for future retrieval
index.storage_context.persist("./storage")
