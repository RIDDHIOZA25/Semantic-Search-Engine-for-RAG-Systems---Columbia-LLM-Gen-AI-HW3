"""
Task A: Document Processing and Vector Store Setup

This module implements:
1. Loading text documents using appropriate document loaders
2. Splitting documents into chunks using RecursiveCharacterTextSplitter
3. Generating embeddings using sentence-transformers/all-MiniLM-L6-v2
4. Storing embeddings in Chroma vector database with proper metadata
5. Verifying chunk creation and vector store setup

Usage:
    python task_a_document_processing.py

Requirements:
    - langchain, langchain-community, langchain-text-splitters
    - chromadb
    - sentence-transformers
    - pypdf
    - Documents should be in ./data/ directory (PDF, TXT, or MD files)

Output:
    - Creates vector store in ./chroma_rag_demo/
    - Displays chunk statistics and verification results
"""

import os
import glob
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
# Try to use newer langchain-huggingface if available, fallback to deprecated version
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    # Fallback to deprecated version (still works, but shows warning)
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


# Configuration constants
DATA_DIR = "./data"
PERSIST_DIR = "./chroma_rag_demo"
COLLECTION = "rag_demo"
CHUNK_SIZE = 1000  # Default chunk size in characters
CHUNK_OVERLAP = 200  # Overlap between chunks in characters
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_documents(data_dir: str = DATA_DIR) -> List[Document]:
    """
    Load documents from the data directory.
    Supports PDF, TXT, and MD file formats.
    
    Args:
        data_dir: Path to directory containing documents
        
    Returns:
        List of Document objects loaded from files
    """
    # Find all document files
    paths = sorted(
        glob.glob(os.path.join(data_dir, "**/*.pdf"), recursive=True)
        + glob.glob(os.path.join(data_dir, "**/*.txt"), recursive=True)
        + glob.glob(os.path.join(data_dir, "**/*.md"), recursive=True)
    )
    
    print(f"\n{'='*60}")
    print(f"Loading Documents from: {data_dir}")
    print(f"{'='*60}")
    print(f"Found {len(paths)} files")
    
    if len(paths) == 0:
        raise ValueError(f"No documents found in {data_dir}. Please ensure files exist.")
    
    docs = []
    for file_path in paths:
        try:
            if file_path.lower().endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                print(f"  Loading PDF: {os.path.basename(file_path)}")
            else:
                loader = TextLoader(file_path, encoding="utf-8")
                file_ext = os.path.splitext(file_path)[1]
                print(f"  Loading {file_ext.upper()}: {os.path.basename(file_path)}")
            
            loaded_docs = loader.load()
            docs.extend(loaded_docs)
            print(f"    → Loaded {len(loaded_docs)} document(s)")
            
        except Exception as e:
            print(f"    ✗ Error loading {file_path}: {e}")
            continue
    
    print(f"\nTotal: {len(docs)} documents loaded from {len(paths)} files")
    if len(docs) > 0:
        total_chars = sum(len(doc.page_content) for doc in docs)
        print(f"Total characters: {total_chars:,}")
        print(f"Average document length: {total_chars // len(docs):,} characters")
    
    return docs


def split_documents(
    docs: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[Document]:
    """
    Split documents into chunks using RecursiveCharacterTextSplitter.
    
    Args:
        docs: List of Document objects to split
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between chunks in characters
        
    Returns:
        List of chunked Document objects
    """
    print(f"\n{'='*60}")
    print("Splitting Documents into Chunks")
    print(f"{'='*60}")
    print(f"Chunk size: {chunk_size} characters")
    print(f"Chunk overlap: {chunk_overlap} characters")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,  # Preserve chunk location metadata
    )
    
    chunks = splitter.split_documents(docs)
    
    print(f"\nCreated {len(chunks)} chunks from {len(docs)} documents")
    
    if len(chunks) > 0:
        # Display chunk statistics
        chunk_lengths = [len(chunk.page_content) for chunk in chunks]
        print(f"\nChunk Statistics:")
        print(f"  Min length: {min(chunk_lengths)} characters")
        print(f"  Max length: {max(chunk_lengths)} characters")
        print(f"  Average length: {sum(chunk_lengths) / len(chunk_lengths):.1f} characters")
        
        # Show sample chunk info
        print(f"\nSample Chunk (First):")
        print(f"  Source: {chunks[0].metadata.get('source', 'N/A')}")
        if 'start_index' in chunks[0].metadata:
            print(f"  Start index: {chunks[0].metadata['start_index']}")
        print(f"  Length: {len(chunks[0].page_content)} characters")
        print(f"  Preview: {chunks[0].page_content[:150]}...")
    
    return chunks


def initialize_embeddings(model_name: str = EMBEDDING_MODEL_NAME):
    """
    Initialize the embedding model using HuggingFace sentence transformers.
    
    Args:
        model_name: Name of the HuggingFace model to use
        
    Returns:
        HuggingFaceEmbeddings instance
    """
    print(f"\n{'='*60}")
    print("Initializing Embedding Model")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},  # Use CPU to avoid PyTorch DLL issues
            encode_kwargs={'normalize_embeddings': False}
        )
        
        # Test the embedding model
        test_embedding = embeddings.embed_query("Test embedding")
        print(f"✓ Embedding model initialized successfully!")
        print(f"  - Embedding dimension: {len(test_embedding)}")
        print(f"  - Provider: HuggingFace Sentence Transformers")
        
        return embeddings
        
    except Exception as e:
        print(f"✗ Error initializing embeddings: {e}")
        raise


def create_vector_store(
    chunks: List[Document],
    embeddings,
    persist_directory: str = PERSIST_DIR,
    collection_name: str = COLLECTION
) -> Chroma:
    """
    Create and persist a Chroma vector store from document chunks.
    
    Args:
        chunks: List of chunked Document objects
        embeddings: Embedding model instance
        persist_directory: Directory to persist the vector store
        collection_name: Name for the Chroma collection
        
    Returns:
        Chroma vector store instance
    """
    print(f"\n{'='*60}")
    print("Creating Vector Store")
    print(f"{'='*60}")
    
    if len(chunks) == 0:
        raise ValueError("Cannot create vector store: no chunks available.")
    
    print(f"Chunks to process: {len(chunks)}")
    print(f"Persist directory: {os.path.abspath(persist_directory)}")
    print(f"Collection name: {collection_name}")
    
    # Create vector store
    # Note: Chroma 0.4.x+ automatically persists, so persist() is not needed
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    
    # Chroma auto-persists, but we can explicitly persist for older versions compatibility
    # (This may show a deprecation warning but won't cause errors)
    try:
        vector_store.persist()
    except AttributeError:
        # Chroma 0.4.x+ doesn't have persist() method (auto-persists)
        pass
    
    print(f"\n✓ Vector store created and persisted successfully!")
    print(f"  - Persisted at: {os.path.abspath(persist_directory)}")
    print(f"  - Total vectors: {vector_store._collection.count()}")
    print(f"  - Collection: {collection_name}")
    
    return vector_store


def verify_vector_store(vector_store: Chroma, num_samples: int = 5):
    """
    Verify the vector store contents and display sample metadata.
    
    Args:
        vector_store: Chroma vector store instance
        num_samples: Number of sample documents to display
    """
    print(f"\n{'='*60}")
    print("Verifying Vector Store")
    print(f"{'='*60}")
    
    total_count = vector_store._collection.count()
    print(f"Total vectors in collection: {total_count}")
    
    # Get sample documents with metadata
    sample_results = vector_store._collection.get(
        include=["metadatas", "documents"],
        limit=num_samples
    )
    
    sample_metadatas = sample_results.get("metadatas", [])
    sample_documents = sample_results.get("documents", [])
    
    print(f"\nSample Documents (showing {len(sample_metadatas)} of {total_count}):")
    for i, (meta, doc) in enumerate(zip(sample_metadatas, sample_documents), 1):
        print(f"\n  Document {i}:")
        print(f"    Source: {meta.get('source', 'N/A')}")
        if 'start_index' in meta:
            print(f"    Start index: {meta['start_index']}")
        print(f"    Length: {len(doc)} characters")
        print(f"    Preview: {doc[:150]}...")
    
    print(f"\n✓ Vector store verification complete!")
    print(f"✓ Task A: Document Processing and Vector Store Setup - COMPLETE")


def main():
    """
    Main function to execute Task A: Document Processing and Vector Store Setup
    """
    print("\n" + "="*60)
    print("TASK A: Document Processing and Vector Store Setup")
    print("="*60)
    
    try:
        # Step 1: Load documents
        docs = load_documents()
        
        # Step 2: Split documents into chunks
        chunks = split_documents(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        
        # Step 3: Initialize embeddings
        embeddings = initialize_embeddings()
        
        # Step 4: Create vector store
        vector_store = create_vector_store(chunks, embeddings)
        
        # Step 5: Verify setup
        verify_vector_store(vector_store)
        
        print(f"\n{'='*60}")
        print("SUCCESS: Task A completed successfully!")
        print(f"{'='*60}")
        
        return vector_store, embeddings
        
    except Exception as e:
        print(f"\n✗ Error in Task A: {e}")
        raise


if __name__ == "__main__":
    vector_store, embeddings = main()

