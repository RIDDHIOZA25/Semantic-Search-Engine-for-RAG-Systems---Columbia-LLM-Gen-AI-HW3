"""
Task B: Implement Semantic Search with Multiple Methods

This module implements:
1. Similarity search using vector_store.similarity_search() method
2. Maximum Marginal Relevance (MMR) search using max_marginal_relevance_search() with λ=0.5
3. Similarity search with scores using similarity_search_with_score()
4. Comparison of all search methods with the same queries

MMR Definition:
A retrieval strategy that balances relevance with diversity to reduce redundant results.
MMR iteratively selects documents that are both relevant to the query AND different from 
already-selected documents.

Formula: MMR = λ × Sim(doc, query) - (1-λ) × max(Sim(doc, selected))
where λ = 0.5

Usage:
    python task_b_semantic_search.py

Prerequisites:
    - Run task_a_document_processing.py first to create the vector store
    - Vector store should exist in ./chroma_rag_demo/
"""

import os
import time
from typing import List, Tuple, Dict
from collections import defaultdict

from langchain_community.vectorstores import Chroma
# Try to use newer langchain-huggingface if available, fallback to deprecated version
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


# Configuration constants
PERSIST_DIR = "./chroma_rag_demo"
COLLECTION = "rag_demo"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MMR_LAMBDA = 0.5  # λ parameter for MMR (balances relevance vs diversity)

# Test queries covering different topics
TEST_QUERIES = [
    "artificial intelligence applications",
    "climate change effects",
    "Olympic games history",
    "economic policy impacts"
]


def load_vector_store(
    persist_directory: str = PERSIST_DIR,
    collection_name: str = COLLECTION
) -> Tuple[Chroma, HuggingFaceEmbeddings]:
    """
    Load the existing vector store and embeddings from Task A.
    
    Args:
        persist_directory: Directory where vector store is persisted
        collection_name: Name of the Chroma collection
        
    Returns:
        Tuple of (Chroma vector store, Embeddings instance)
    """
    print(f"\n{'='*60}")
    print("Loading Vector Store")
    print(f"{'='*60}")
    
    if not os.path.exists(persist_directory):
        raise ValueError(
            f"Vector store not found at {persist_directory}. "
            "Please run task_a_document_processing.py first."
        )
    
    # Initialize embeddings (same as Task A)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )
    
    # Load vector store
    vector_store = Chroma(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_function=embeddings
    )
    
    total_vectors = vector_store._collection.count()
    print(f"✓ Vector store loaded successfully!")
    print(f"  - Location: {os.path.abspath(persist_directory)}")
    print(f"  - Collection: {collection_name}")
    print(f"  - Total vectors: {total_vectors}")
    
    return vector_store, embeddings


def similarity_search(
    vector_store: Chroma,
    query: str,
    k: int = 5
) -> List[Document]:
    """
    Perform similarity search using cosine similarity.
    Returns the k most similar documents to the query.
    
    Args:
        vector_store: Chroma vector store instance
        query: Search query string
        k: Number of documents to retrieve
        
    Returns:
        List of Document objects (most similar first)
    """
    results = vector_store.similarity_search(query, k=k)
    return results


def similarity_search_with_score(
    vector_store: Chroma,
    query: str,
    k: int = 5
) -> List[Tuple[Document, float]]:
    """
    Perform similarity search and return documents with similarity scores.
    Lower scores indicate higher similarity (distance metric).
    
    Args:
        vector_store: Chroma vector store instance
        query: Search query string
        k: Number of documents to retrieve
        
    Returns:
        List of tuples: (Document, similarity_score)
        Note: Lower score = more similar (distance metric)
    """
    results = vector_store.similarity_search_with_score(query, k=k)
    return results


def mmr_search(
    vector_store: Chroma,
    query: str,
    k: int = 5,
    lambda_param: float = MMR_LAMBDA,
    fetch_k: int = 20
) -> List[Document]:
    """
    Perform Maximum Marginal Relevance (MMR) search.
    
    MMR balances relevance with diversity:
    - Selects documents that are relevant to the query
    - AND different from already-selected documents
    
    Formula: MMR = λ × Sim(doc, query) - (1-λ) × max(Sim(doc, selected))
    where λ balances relevance (λ high) vs diversity (λ low)
    
    Args:
        vector_store: Chroma vector store instance
        query: Search query string
        k: Number of documents to retrieve
        lambda_param: λ parameter (0.0 to 1.0)
                     - λ=1.0: Pure relevance (similar to similarity_search)
                     - λ=0.0: Pure diversity
                     - λ=0.5: Balanced (default)
        fetch_k: Number of documents to fetch before applying MMR
                 (larger values improve diversity but slower)
        
    Returns:
        List of Document objects selected by MMR algorithm
    """
    results = vector_store.max_marginal_relevance_search(
        query,
        k=k,
        lambda_mult=lambda_param,
        fetch_k=fetch_k
    )
    return results


def display_search_results(
    query: str,
    similarity_results: List[Document],
    similarity_with_scores: List[Tuple[Document, float]],
    mmr_results: List[Document],
    method_names: Tuple[str, str, str] = ("Similarity Search", "Similarity with Scores", "MMR Search")
):
    """
    Display and compare results from different search methods.
    
    Args:
        query: The search query
        similarity_results: Results from similarity_search()
        similarity_with_scores: Results from similarity_search_with_score()
        mmr_results: Results from mmr_search()
        method_names: Names for the three methods
    """
    print(f"\n{'='*80}")
    print(f"QUERY: {query}")
    print(f"{'='*80}")
    
    # Method 1: Similarity Search
    print(f"\n{method_names[0]} (Top {len(similarity_results)} results):")
    print("-" * 80)
    for i, doc in enumerate(similarity_results, 1):
        source = doc.metadata.get('source', 'N/A')
        preview = doc.page_content[:200].replace('\n', ' ')
        print(f"\n  [{i}] Source: {os.path.basename(source)}")
        print(f"      Preview: {preview}...")
    
    # Method 2: Similarity Search with Scores
    print(f"\n{method_names[1]} (Top {len(similarity_with_scores)} results with scores):")
    print("-" * 80)
    print("  (Lower score = more similar)")
    for i, (doc, score) in enumerate(similarity_with_scores, 1):
        source = doc.metadata.get('source', 'N/A')
        preview = doc.page_content[:200].replace('\n', ' ')
        print(f"\n  [{i}] Score: {score:.4f}")
        print(f"      Source: {os.path.basename(source)}")
        print(f"      Preview: {preview}...")
    
    # Method 3: MMR Search
    print(f"\n{method_names[2]} (Top {len(mmr_results)} results, λ={MMR_LAMBDA}):")
    print("-" * 80)
    for i, doc in enumerate(mmr_results, 1):
        source = doc.metadata.get('source', 'N/A')
        preview = doc.page_content[:200].replace('\n', ' ')
        print(f"\n  [{i}] Source: {os.path.basename(source)}")
        print(f"      Preview: {preview}...")
    
    # Compare: Show which documents are unique to each method
    print(f"\n{'─'*80}")
    print("COMPARISON: Document Sources")
    print(f"{'─'*80}")
    
    sim_sources = {os.path.basename(d.metadata.get('source', '')) for d in similarity_results}
    mmr_sources = {os.path.basename(d.metadata.get('source', '')) for d in mmr_results}
    
    print(f"Similarity Search unique sources: {sim_sources - mmr_sources}")
    print(f"MMR Search unique sources: {mmr_sources - sim_sources}")
    print(f"Common sources: {sim_sources & mmr_sources}")
    print(f"Diversity: Similarity={len(sim_sources)} unique sources, MMR={len(mmr_sources)} unique sources")


def compare_search_methods(
    vector_store: Chroma,
    queries: List[str] = TEST_QUERIES,
    k: int = 5
):
    """
    Compare all three search methods with multiple queries.
    
    Args:
        vector_store: Chroma vector store instance
        queries: List of query strings to test
        k: Number of documents to retrieve per query
    """
    print(f"\n{'='*80}")
    print("TASK B: Semantic Search Methods Comparison")
    print(f"{'='*80}")
    print(f"Testing {len(queries)} queries with k={k} results each")
    print(f"MMR λ parameter: {MMR_LAMBDA}")
    
    for query in queries:
        # Perform all three search methods
        similarity_results = similarity_search(vector_store, query, k=k)
        similarity_with_scores = similarity_search_with_score(vector_store, query, k=k)
        mmr_results = mmr_search(vector_store, query, k=k, lambda_param=MMR_LAMBDA)
        
        # Display comparison
        display_search_results(query, similarity_results, similarity_with_scores, mmr_results)


def measure_search_latency(
    vector_store: Chroma,
    query: str,
    k_values: List[int] = [1, 5, 10],
    num_runs: int = 3
) -> Dict[str, Dict[int, float]]:
    """
    Measure search latency for different k values.
    
    Args:
        vector_store: Chroma vector store instance
        query: Test query
        k_values: List of k values to test
        num_runs: Number of runs to average
        
    Returns:
        Dictionary mapping method names to k->average_latency
    """
    print(f"\n{'='*80}")
    print(f"Latency Measurement for Query: '{query}'")
    print(f"{'='*80}")
    
    results = {
        "Similarity Search": {},
        "Similarity with Scores": {},
        "MMR Search": {}
    }
    
    for k in k_values:
        # Similarity Search
        times = []
        for _ in range(num_runs):
            start = time.time()
            similarity_search(vector_store, query, k=k)
            times.append(time.time() - start)
        results["Similarity Search"][k] = sum(times) / len(times)
        
        # Similarity Search with Scores
        times = []
        for _ in range(num_runs):
            start = time.time()
            similarity_search_with_score(vector_store, query, k=k)
            times.append(time.time() - start)
        results["Similarity with Scores"][k] = sum(times) / len(times)
        
        # MMR Search
        times = []
        for _ in range(num_runs):
            start = time.time()
            mmr_search(vector_store, query, k=k)
            times.append(time.time() - start)
        results["MMR Search"][k] = sum(times) / len(times)
    
    # Display results
    print(f"\n{'Method':<30} {'k=1':<12} {'k=5':<12} {'k=10':<12}")
    print("-" * 80)
    for method, latencies in results.items():
        print(f"{method:<30} {latencies[1]*1000:>8.3f}ms  {latencies[5]*1000:>8.3f}ms  {latencies[10]*1000:>8.3f}ms")
    
    return results


def demonstrate_mmr_advantage(vector_store: Chroma, query: str, k: int = 5):
    """
    Demonstrate scenarios where MMR provides better coverage than similarity search.
    Shows how MMR reduces redundancy by selecting diverse documents.
    
    Args:
        vector_store: Chroma vector store instance
        query: Test query
        k: Number of documents to retrieve
    """
    print(f"\n{'='*80}")
    print("MMR vs Similarity Search: Diversity Analysis")
    print(f"{'='*80}")
    print(f"Query: '{query}'")
    
    similarity_results = similarity_search(vector_store, query, k=k)
    mmr_results = mmr_search(vector_store, query, k=k, lambda_param=MMR_LAMBDA)
    
    # Analyze source diversity
    sim_sources = defaultdict(int)
    mmr_sources = defaultdict(int)
    
    for doc in similarity_results:
        source = os.path.basename(doc.metadata.get('source', 'unknown'))
        sim_sources[source] += 1
    
    for doc in mmr_results:
        source = os.path.basename(doc.metadata.get('source', 'unknown'))
        mmr_sources[source] += 1
    
    print(f"\nSimilarity Search Source Distribution (Top {k} results):")
    for source, count in sorted(sim_sources.items(), key=lambda x: -x[1]):
        print(f"  {source}: {count} chunk(s)")
    print(f"  Unique sources: {len(sim_sources)}")
    
    print(f"\nMMR Search Source Distribution (Top {k} results, λ={MMR_LAMBDA}):")
    for source, count in sorted(mmr_sources.items(), key=lambda x: -x[1]):
        print(f"  {source}: {count} chunk(s)")
    print(f"  Unique sources: {len(mmr_sources)}")
    
    # Check if MMR provides better diversity
    if len(mmr_sources) > len(sim_sources):
        print(f"\n✓ MMR provides better diversity: {len(mmr_sources)} vs {len(sim_sources)} unique sources")
    elif len(mmr_sources) == len(sim_sources):
        print(f"\n→ Both methods found {len(mmr_sources)} unique sources")
    else:
        print(f"\n→ Similarity search found more unique sources ({len(sim_sources)} vs {len(mmr_sources)})")
    
    # Show redundant documents in similarity search
    redundant = [(source, count) for source, count in sim_sources.items() if count > 1]
    if redundant:
        print(f"\nSimilarity Search redundancy: {len(redundant)} source(s) appear multiple times")
        for source, count in redundant:
            print(f"  {source}: appears {count} times")


def main():
    """
    Main function to execute Task B: Semantic Search Methods
    """
    try:
        # Load vector store from Task A
        vector_store, embeddings = load_vector_store()
        
        # Compare all search methods with test queries
        compare_search_methods(vector_store, queries=TEST_QUERIES, k=5)
        
        # Measure latency for different k values
        print(f"\n{'='*80}")
        print("LATENCY MEASUREMENTS")
        print(f"{'='*80}")
        measure_search_latency(vector_store, TEST_QUERIES[0], k_values=[1, 5, 10])
        
        # Demonstrate MMR advantage
        demonstrate_mmr_advantage(vector_store, TEST_QUERIES[0], k=5)
        
        print(f"\n{'='*80}")
        print("SUCCESS: Task B completed successfully!")
        print(f"{'='*80}")
        print("\nSummary:")
        print("  ✓ Similarity search implemented")
        print("  ✓ Similarity search with scores implemented")
        print("  ✓ MMR search implemented (λ=0.5)")
        print("  ✓ All methods demonstrated and compared")
        
    except Exception as e:
        print(f"\n✗ Error in Task B: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

