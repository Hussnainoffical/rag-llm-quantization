# Retrieval-Augmented Generation (RAG) Using Unsloth Dynamic 4-bit LLMs

This repository contains an end-to-end implementation of a **Retrieval-Augmented Generation (RAG) pipeline** built using **Unsloth dynamic 4-bit quantized large language models**.  
The project focuses on efficient document retrieval, grounded text generation, and optimized GPU memory usage, making it suitable for execution in **resource-constrained environments such as Google Colab**.

---

## Project Overview

Traditional language models rely only on their internal knowledge, which can lead to outdated or hallucinated responses.  
This project addresses that limitation by combining **document retrieval** with **large language model generation**, ensuring responses are grounded in external knowledge sources.

To further improve efficiency, the system uses **Unsloth’s dynamic 4-bit quantization**, which significantly reduces VRAM usage while preserving model accuracy.

---

## Architecture Overview

1. **Document Ingestion**
   - Domain-specific documents are loaded and preprocessed
   - Text is split into manageable chunks for efficient retrieval

2. **Embedding and Indexing**
   - Text chunks are converted into vector embeddings
   - FAISS is used as a vector database for fast similarity search

3. **Query Processing**
   - User query is embedded
   - Top-k relevant document chunks are retrieved from the vector store

4. **Grounded Generation**
   - Retrieved context is injected into the prompt
   - A dynamic 4-bit quantized Unsloth LLM generates a grounded response

5. **Memory Optimization**
   - Dynamic quantization preserves higher precision for critical parameters
   - Enables efficient inference on limited GPU memory

---

## Technologies Used

- **Python**
- **Unsloth (Dynamic 4-bit Quantization)**
- **Hugging Face Transformers**
- **FAISS (Vector Search)**
- **Sentence Embeddings**
- **PyTorch**
- **Google Colab (GPU Execution)**

---

## Why Unsloth Dynamic 4-bit Quantization

- Reduces VRAM usage significantly compared to full-precision models
- Preserves accuracy by selectively maintaining higher precision parameters
- Allows large language models to run efficiently on limited GPUs
- Ideal for scalable and cost-effective RAG systems

---

## Key Features

- End-to-end RAG pipeline
- Grounded and context-aware text generation
- Low-VRAM optimized inference
- Modular and easy-to-extend design
- Suitable for academic, research, and internship-level projects

---

## How to Run

1. Open the notebook in **Google Colab**
2. Enable GPU runtime
3. Install required dependencies
4. Run cells sequentially to:
   - Load documents
   - Build the vector index
   - Perform retrieval
   - Generate grounded responses

---

## Use Cases

- Question answering over custom documents
- Knowledge-based chat systems
- Research assistants
- Efficient enterprise LLM applications

---


Developed as part of an **AI internship project**, focusing on real-world LLM optimization and efficient AI system design.
