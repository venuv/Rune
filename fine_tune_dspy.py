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
custom_rag = RAG(query_engine)

# Define another Signature class for assessing the quality of generated answers
class Assess(dspy.Signature):
    """Evaluates the quality of QA answers along specified dimensions."""
    assessed_text = dspy.InputField()
    assessment_question = dspy.InputField()
    assessment_answer = dspy.OutputField(desc="Yes or No")

# Define a metric function for validating generated answers
def llm_validate_metric(gold, pred, trace=None):
    """Calculates a validation score for predicted answers against gold standard answers."""
    question, answer, actual_answer = gold.question, gold.answer, pred.answer
    # Assessment questions for evaluating the answer
    engaging = "Does the assessed text make for a self-contained and engaging answer?"
    correct = f"The text should answer `{question}` with `{answer}`. Does the assessed text contain this answer?"

    # Context manager for using the specified language model
    with dspy.context(lm=gpt4T):
        correct = dspy.Predict(Assess)(assessed_text=actual_answer, assessment_question=correct)
        engaging = dspy.Predict(Assess)(assessed_text=actual_answer, assessment_question=engaging)
    # Calculate the final score
    correct, engaging = [m.assessment_answer.lower() == 'yes' for m in [correct, engaging]]
    score = (correct + engaging) if correct else 0
    return score / 2.0 if not trace else score >= 2

# Sample QA pairs for training the model
qa_pair1 = dspy.Example(question="In 1997, Why does Buffett think Berkshire look-through earnings are a better reflection of economic progress than reported earnings?",
                        answer="Buffett argues that look-through earnings, which include Berkshire share of the retained earnings from its investee companies, more accurately capture the value being created at Berkshire than just the reported earnings that only include dividends received from investees.")

qa_pair2 = dspy.Example(question="In 2009, How does Warren Buffett describe Berkshire approach to acquisitions?",
                        answer="Buffett emphasizes the importance of looking for large acquisitions with consistent earning power, good returns on equity, simple businesses, and existing management. He values confidentiality, quick responses, and a preference for cash transactions.")

qa_pair3 = dspy.Example(question="In 2017, what is the main focus of Berkshire Hathaway's management in terms of evaluating performance?",
                        answer="Berkshire Hathaway's management focuses on increases in normalized per-share earning power as the key metric for evaluating performance.")

qa_pair4 = dspy.Example(question="In 1985, How does Warren Buffett view the relationship between Berkshire's market value and business value?",
                        answer="Warren Buffett prefers a market price that consistently approximates business value to ensure that all owners prosper as the business prospers during their ownership period. He emphasizes that wild swings in market prices above or below business value do not change the final gains for owners in aggregate")

qa_pair5 = dspy.Example(question="In 1997, What are Buffett's thoughts on Berkshire's insurance operations and the concept of \"float\"?",
                        answer="Buffett views Berkshire's insurance operations, particularly the ability to generate low-cost or better-than-free \"float\" from premiums held before paying out losses, as a key competitive advantage that has compounded value over decades.")

# Tell DSPy that the 'question' field is the input
trainset = [
    qa_pair1.with_inputs('question'),
    qa_pair2.with_inputs('question'),
    qa_pair3.with_inputs('question'),
    qa_pair4.with_inputs('question'),
    qa_pair5.with_inputs('question'),
]

# Compile the RAG module using the BootstrapFewShot teleprompter with the defined metric
teleprompter = BootstrapFewShot(metric=llm_validate_metric)
compiled_rag = teleprompter.compile(custom_rag, trainset=trainset)
compiled_rag.save("buffett_dspy_model")

# Demonstrate using the compiled RAG module to answer a question
question = "Why does Buffett think Berkshire's look-through earnings are a better reflection of economic progress than reported earnings?"
pred = custom_rag(question)  # Predict using the custom RAG module
pred_compiled = compiled_rag(question)  # Predict using the compiled RAG module
print(f"Question: {question}")
print(f"Predicted Answer: {pred.answer}")
print(f"Predicted Fine-Tuned Answer: {pred_compiled.answer}")
