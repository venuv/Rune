# Import necessary libraries and modules for processing and evaluating text with DSPy and LLaMA
import os
import dspy
from dspy import Signature, InputField, OutputField, Module, Predict, Prediction

# LLaMA parsing and indexing utilities for document processing
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
# DSPy utilities for model training and evaluation
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate import answer_exact_match, answer_passage_match
from dspy import Example
from dotenv import load_dotenv

load_dotenv()

# Securely set the OpenAI API key as an environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")
llamaparse_api_key = os.getenv("LLAMAPARSE_API_KEY")

# Error handling for missing keys (optional)
if not openai_api_key or not llamaparse_api_key:
    raise ValueError("Missing required environment variables: OPENAI_API_KEY, LLAMAPARSE_API_KEY")


# Configure language models with the DSPy framework
turbo = dspy.OpenAI(model='gpt-3.5-turbo',max_tokens=4096)
gpt4T = dspy.OpenAI(model='gpt-4-1106-preview', max_tokens=1000, model_type='chat')
dspy.settings.configure(lm=turbo)


# Initialize storage context for loading or persisting indexes
storage_context = StorageContext.from_defaults(persist_dir="storage")

# Load an existing vector index from storage and prepare it for querying
index = load_index_from_storage(storage_context, index_id="vector_index")
query_engine = index.as_query_engine(response_mode="tree_summarize")

# Define a Signature class to outline the input and output fields for generating answers
class GenerateAnswer(dspy.Signature):
    """Answer questions from 40 years worth of Buffetts investment newsletters in ways useful to a seasoned stock investor"""
    context = dspy.InputField(desc="may contain most relevant passages for the question being asked")
    question = dspy.InputField(desc="the text of the question")
    answer = dspy.OutputField(desc="one to two paragraphs. should contain the answer, the rationale, and any supporting documentation")

# Define a DSPy Module to utilize the Retrieve and Generate pattern for answering questions
class RAG(dspy.Module):
    """Retrieval-Augmented Generation module for querying and generating responses."""
    def __init__(self, num_passages=3):
        super().__init__()
        self.query_engine = query_engine
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question):
        """Generates an answer to a question by querying a document index and synthesizing information."""
        response = self.query_engine.query(question)
        context = response.response
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)

# Instantiate the RAG module with the query engine
compiled_rag = RAG(query_engine)
compiled_rag.load("buffett_dspy_model")

# Demonstrate using the compiled RAG module to answer a question
question = "Why does Buffett think Berkshire's look-through earnings are a better reflection of economic progress than reported earnings?"
pred_compiled = compiled_rag(question)  # Predict using the compiled RAG module
print(f"Question: {question}")
print(f"Compiled Buffett Model Answer: {pred_compiled.answer}")
