"""
Bonus 1: Hybrid Search - Combining Semantic Search with BM25 Keyword Scoring

This module implements a hybrid search that combines:
- Semantic search (vector similarity)
- BM25 keyword-based scoring

The hybrid approach merges results from both methods using weighted scoring.

Usage:
    python bonus_1_hybrid_search.py

Prerequisites:
    - Run task_a_document_processing.py first to create vector store
    - Requires: rank-bm25 package (pip install rank-bm25)
"""

import os
import numpy as np
from typing import List, Tuple, Dict
from collections import Counter

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError("Please install rank-bm25: pip install rank-bm25")

from langchain_community.vectorstores import Chroma
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from task_b_semantic_search import (
    similarity_search_with_score,
    mmr_search,
    TEST_QUERIES,
    PERSIST_DIR,
    COLLECTION,
    EMBEDDING_MODEL_NAME
)
from langchain_core.documents import Document


class HybridSearch:
    """
    Hybrid search combining semantic search with BM25 keyword matching.
    """
    
    def __init__(self, vector_store: Chroma, documents: List[Document]):
        """
        Initialize hybrid search with vector store and documents.
        
        Args:
            vector_store: Chroma vector store for semantic search
            documents: List of documents for BM25 indexing
        """
        self.vector_store = vector_store
        self.documents = documents
        
        # Prepare BM25 index
        # Tokenize documents for BM25
        tokenized_docs = [self._tokenize(doc.page_content) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        print(f"✓ Hybrid search initialized")
        print(f"  - Vector store: {vector_store._collection.count()} documents")
        print(f"  - BM25 index: {len(documents)} documents")
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization (can be improved with nltk).
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        return text.lower().split()
    
    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        semantic_weight: float = 0.7,
        bm25_weight: float = 0.3,
        use_mmr: bool = False
    ) -> List[Tuple[Document, float]]:
        """
        Perform hybrid search combining semantic and BM25.
        
        Args:
            query: Search query
            k: Number of results to return
            semantic_weight: Weight for semantic search scores (0.0 to 1.0)
            bm25_weight: Weight for BM25 scores (0.0 to 1.0)
            use_mmr: Whether to use MMR for semantic search (default: False, uses similarity)
            
        Returns:
            List of (Document, hybrid_score) tuples, sorted by score descending
        """
        # Normalize weights
        total_weight = semantic_weight + bm25_weight
        semantic_weight /= total_weight
        bm25_weight /= total_weight
        
        # 1. Semantic search - get scored results for combining
        scored_results = similarity_search_with_score(self.vector_store, query, k=k*2)
        
        # 2. BM25 search
        query_tokens = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        # Normalize BM25 scores to 0-1 range
        if len(bm25_scores) > 0:
            max_bm25 = max(bm25_scores)
            min_bm25 = min(bm25_scores)
            if max_bm25 > min_bm25:
                bm25_scores = [(s - min_bm25) / (max_bm25 - min_bm25) for s in bm25_scores]
            else:
                bm25_scores = [0.0] * len(bm25_scores)
        
        # 3. Combine scores
        # Create a mapping from document content hash to index
        doc_content_to_index = {}
        for idx, doc in enumerate(self.documents):
            # Use a hash of content + source as unique identifier
            doc_key = hash((doc.page_content[:100], doc.metadata.get('source', '')))
            doc_content_to_index[doc_key] = idx
        
        # Map semantic results to document indices with scores
        semantic_to_index = {}
        for doc, score in scored_results:
            doc_key = hash((doc.page_content[:100], doc.metadata.get('source', '')))
            if doc_key in doc_content_to_index:
                idx = doc_content_to_index[doc_key]
                # Convert distance to similarity score
                semantic_to_index[idx] = 1.0 / (1.0 + score)
        
        # Calculate hybrid scores for all documents
        hybrid_results = []
        for idx, doc in enumerate(self.documents):
            semantic_score = semantic_to_index.get(idx, 0.0)
            bm25_score = bm25_scores[idx] if idx < len(bm25_scores) else 0.0
            
            hybrid_score = (semantic_weight * semantic_score) + (bm25_weight * bm25_score)
            hybrid_results.append((doc, hybrid_score, semantic_score, bm25_score))
        
        # Sort by hybrid score and return top k
        hybrid_results.sort(key=lambda x: x[1], reverse=True)
        return hybrid_results[:k]
    
    def compare_search_methods(
        self,
        query: str,
        k: int = 5
    ):
        """
        Compare semantic, BM25, and hybrid search results.
        
        Args:
            query: Search query
            k: Number of results
        """
        print(f"\n{'='*80}")
        print(f"Hybrid Search Comparison for Query: '{query}'")
        print(f"{'='*80}")
        
        # Semantic search
        semantic_results = similarity_search_with_score(self.vector_store, query, k=k)
        
        # BM25 search
        query_tokens = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(query_tokens)
        bm25_indices = np.argsort(bm25_scores)[::-1][:k]
        bm25_results = [
            (self.documents[idx], bm25_scores[idx])
            for idx in bm25_indices
        ]
        
        # Hybrid search
        hybrid_results = self.hybrid_search(query, k=k, semantic_weight=0.7, bm25_weight=0.3)
        
        # Display results
        print(f"\nSemantic Search (Top {k}):")
        print("-" * 80)
        for i, (doc, score) in enumerate(semantic_results, 1):
            source = os.path.basename(doc.metadata.get('source', 'unknown'))
            print(f"  [{i}] Semantic Score: {1.0/(1.0+score):.4f} | Source: {source}")
            print(f"      Preview: {doc.page_content[:150]}...")
        
        print(f"\nBM25 Keyword Search (Top {k}):")
        print("-" * 80)
        for i, (doc, score) in enumerate(bm25_results, 1):
            source = os.path.basename(doc.metadata.get('source', 'unknown'))
            print(f"  [{i}] BM25 Score: {score:.4f} | Source: {source}")
            print(f"      Preview: {doc.page_content[:150]}...")
        
        print(f"\nHybrid Search (Semantic 70% + BM25 30%, Top {k}):")
        print("-" * 80)
        for i, (doc, hybrid_score, sem_score, bm25_score) in enumerate(hybrid_results, 1):
            source = os.path.basename(doc.metadata.get('source', 'unknown'))
            print(f"  [{i}] Hybrid: {hybrid_score:.4f} (Sem: {sem_score:.4f}, BM25: {bm25_score:.4f})")
            print(f"      Source: {source}")
            print(f"      Preview: {doc.page_content[:150]}...")
        
        # Source diversity comparison
        sem_sources = {os.path.basename(d.metadata.get('source', '')) for d, _ in semantic_results}
        bm25_sources = {os.path.basename(d.metadata.get('source', '')) for d, _ in bm25_results}
        hybrid_sources = {os.path.basename(d.metadata.get('source', '')) for d, _, _, _ in hybrid_results}
        
        print(f"\n{'─'*80}")
        print("Source Diversity:")
        print(f"  Semantic: {len(sem_sources)} unique sources")
        print(f"  BM25: {len(bm25_sources)} unique sources")
        print(f"  Hybrid: {len(hybrid_sources)} unique sources")


def load_documents_for_hybrid(vector_store: Chroma) -> List[Document]:
    """
    Load all documents from vector store for BM25 indexing.
    
    Args:
        vector_store: Chroma vector store
        
    Returns:
        List of Document objects
    """
    all_data = vector_store._collection.get(include=["metadatas", "documents"])
    
    documents = []
    for metadata, content in zip(all_data['metadatas'], all_data['documents']):
        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)
    
    return documents


def load_vector_store():
    """Load vector store from Task A."""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )
    
    vector_store = Chroma(
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION,
        embedding_function=embeddings
    )
    
    return vector_store


def main():
    """
    Main function to demonstrate hybrid search.
    """
    print("\n" + "="*80)
    print("BONUS 1: Hybrid Search (Semantic + BM25)")
    print("="*80)
    
    try:
        # Load vector store
        print("\nLoading vector store...")
        vector_store = load_vector_store()
        
        # Load documents for BM25
        print("Loading documents for BM25 indexing...")
        documents = load_documents_for_hybrid(vector_store)
        print(f"✓ Loaded {len(documents)} documents")
        
        # Initialize hybrid search
        hybrid_searcher = HybridSearch(vector_store, documents)
        
        # Compare search methods
        for query in TEST_QUERIES:
            hybrid_searcher.compare_search_methods(query, k=5)
        
        print(f"\n{'='*80}")
        print("SUCCESS: Hybrid Search Demonstration Complete!")
        print(f"{'='*80}")
        print("\nKey Features:")
        print("  ✓ Combines semantic search (vector similarity)")
        print("  ✓ Combines BM25 keyword matching")
        print("  ✓ Weighted scoring (configurable semantic/BM25 weights)")
        print("  ✓ Compares all three methods side-by-side")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

