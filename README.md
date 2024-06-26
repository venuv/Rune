# Rune
### What
At an application level, this application is a **RAG based Q&A on top of 40 years of Warren Buffett Newsletters**.

As a pipeline it demonstrates the few shot fine tune of a DSPy RAG model. In particular the examples in the [DSPy Documentation]( https://dspy-docs.vercel.app/) work with Weaviate Retriever (that is DSPy lingo for the storage and indexing model). To make life simpler, this code runs a pdf File-based Retriever built using LlamaIndex.

---
### How
Code has 3 parts to it:
- **populate_vector_db** takes a named directory of pdf files, and uses LlamaParse to store its embeddings in a VectorDB
- **fine_tune_dspy.py** fine tunes a basic RAG atop the Vector DB with various switches (such as using a TreeSummarizer) that can be changed to suit your needs. It also is 'agentic' in that it uses another LLM instance to measure the quality and relevance of the answer, instead of standard IR metrics. Just for grins - it prints out Q&A responses with base and fine-tuned models, so you can vary the fine tune parameters if you wish. It also saves the compiled model into the directory **"buffett_dspy_model**
- **run_compiled_model.py** illustrates loading a fine-tuned model and using it for Q&A
---
### Miscellaneous
2 keys - LLM and LlamaParse. The Buffett Letters are provided in pdf form in a sub-directory. Use the path to this sub-directory in populate_vector_db. The 'gold standard' answers for the few shot BootStrap Learner were generated by a combination of cGPT, Bard and Claude so as to mix it up -- YMMV.


