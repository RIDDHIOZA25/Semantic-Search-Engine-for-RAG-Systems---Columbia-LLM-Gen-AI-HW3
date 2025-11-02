"""
Bonus 2: Compare Different Embedding Models

This module implements and compares different embedding models:
- sentence-transformers/all-MiniLM-L6-v2 (default)
- sentence-transformers/all-mpnet-base-v2 (larger, higher quality)
- Optional: OpenAI embeddings (if API key available)
- Optional: Cohere embeddings (if API key available)

Usage:
    python bonus_2_embedding_comparison.py

Prerequisites:
    - Run task_a_document_processing.py first
    - For OpenAI: pip install openai, set OPENAI_API_KEY
    - For Cohere: pip install cohere, set COHERE_API_KEY
"""

import os
import time
import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict

try:
    from langchain_openai import OpenAIEmbeddings
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from langchain_community.embeddings import CohereEmbeddings
    HAS_COHERE = True
except ImportError:
    HAS_COHERE = False

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma
from task_a_document_processing import load_documents, split_documents, CHUNK_SIZE, CHUNK_OVERLAP
from task_b_semantic_search import similarity_search, TEST_QUERIES


class EmbeddingComparison:
    """Compare different embedding models."""
    
    def __init__(self, documents, chunks):
        """
        Initialize with documents and chunks.
        
        Args:
            documents: Original documents
            chunks: Document chunks
        """
        self.documents = documents
        self.chunks = chunks
        self.models = {}
        self.vector_stores = {}
    
    def setup_model(self, name: str, embeddings, model_name: str):
        """
        Set up an embedding model and create vector store.
        
        Args:
            name: Display name for the model
            embeddings: Embedding model instance
            model_name: Internal name for the model
        """
        print(f"\n{'─'*80}")
        print(f"Setting up: {name}")
        print(f"{'─'*80}")
        
        try:
            # Test embedding
            test_emb = embeddings.embed_query("Test query")
            dim = len(test_emb)
            print(f"✓ Model loaded: {dim}-dimensional embeddings")
            
            # Create vector store
            persist_dir = f"./chroma_bonus_{model_name}"
            if os.path.exists(persist_dir):
                import shutil
                shutil.rmtree(persist_dir)
            
            vector_store = Chroma.from_documents(
                documents=self.chunks,
                embedding=embeddings,
                persist_directory=persist_dir,
                collection_name="comparison"
            )
            
            self.models[name] = embeddings
            self.vector_stores[name] = vector_store
            print(f"✓ Vector store created: {vector_store._collection.count()} vectors")
            
        except Exception as e:
            print(f"✗ Error setting up {name}: {e}")
    
    def setup_all_models(self):
        """Set up all available embedding models."""
        print(f"\n{'='*80}")
        print("Initializing Embedding Models")
        print(f"{'='*80}")
        
        # 1. Sentence Transformers - MiniLM (small, fast)
        embeddings_minilm = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )
        self.setup_model("MiniLM-L6-v2", embeddings_minilm, "minilm")
        
        # 2. Sentence Transformers - MPNet (larger, higher quality)
        embeddings_mpnet = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )
        self.setup_model("MPNet-base-v2", embeddings_mpnet, "mpnet")
        
        # 3. OpenAI embeddings (if available)
        if HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
            try:
                embeddings_openai = OpenAIEmbeddings(model="text-embedding-3-small")
                self.setup_model("OpenAI text-embedding-3-small", embeddings_openai, "openai")
            except Exception as e:
                print(f"⚠ OpenAI setup failed: {e}")
        else:
            print("\n⚠ OpenAI embeddings not available (set OPENAI_API_KEY)")
        
        # 4. Cohere embeddings (if available)
        if HAS_COHERE and os.getenv("COHERE_API_KEY"):
            try:
                embeddings_cohere = CohereEmbeddings(model="embed-english-v3.0")
                self.setup_model("Cohere embed-english-v3.0", embeddings_cohere, "cohere")
            except Exception as e:
                print(f"⚠ Cohere setup failed: {e}")
        else:
            print("\n⚠ Cohere embeddings not available (set COHERE_API_KEY)")
    
    def compare_models(self, query: str, k: int = 5):
        """
        Compare retrieval results across different embedding models.
        
        Args:
            query: Test query
            k: Number of results
        """
        print(f"\n{'='*80}")
        print(f"Embedding Model Comparison for Query: '{query}'")
        print(f"{'='*80}")
        
        results = {}
        
        for model_name, vector_store in self.vector_stores.items():
            # Measure latency
            start = time.time()
            search_results = similarity_search(vector_store, query, k=k)
            latency = (time.time() - start) * 1000
            
            # Analyze results
            sources = [os.path.basename(doc.metadata.get('source', '')) for doc in search_results]
            unique_sources = len(set(sources))
            
            results[model_name] = {
                'results': search_results,
                'sources': sources,
                'unique_sources': unique_sources,
                'latency_ms': latency
            }
        
        # Display comparison
        print(f"\n{'Model':<25} {'Latency (ms)':<15} {'Unique Sources':<15}")
        print("-" * 80)
        for model_name, metrics in results.items():
            print(f"{model_name:<25} {metrics['latency_ms']:<15.2f} {metrics['unique_sources']:<15}")
        
        # Show top results for each model
        print(f"\n{'─'*80}")
        print("Top Results by Model:")
        print(f"{'─'*80}")
        
        for model_name, metrics in results.items():
            print(f"\n{model_name}:")
            for i, doc in enumerate(metrics['results'][:3], 1):
                source = os.path.basename(doc.metadata.get('source', ''))
                preview = doc.page_content[:120].replace('\n', ' ')
                print(f"  [{i}] {source}: {preview}...")
        
        return results
    
    def comprehensive_comparison(self, queries: List[str] = TEST_QUERIES):
        """
        Comprehensive comparison across multiple queries.
        
        Args:
            queries: List of test queries
        """
        print(f"\n{'='*80}")
        print("Comprehensive Embedding Model Comparison")
        print(f"{'='*80}")
        
        summary = defaultdict(lambda: {'total_latency': 0, 'total_unique_sources': 0, 'queries': 0})
        
        for query in queries:
            results = self.compare_models(query, k=5)
            
            for model_name, metrics in results.items():
                summary[model_name]['total_latency'] += metrics['latency_ms']
                summary[model_name]['total_unique_sources'] += metrics['unique_sources']
                summary[model_name]['queries'] += 1
        
        # Summary statistics
        print(f"\n{'='*80}")
        print("Summary Statistics (Averaged Across All Queries)")
        print(f"{'='*80}")
        print(f"\n{'Model':<25} {'Avg Latency (ms)':<20} {'Avg Unique Sources':<20}")
        print("-" * 80)
        
        for model_name, stats in summary.items():
            avg_latency = stats['total_latency'] / stats['queries']
            avg_sources = stats['total_unique_sources'] / stats['queries']
            print(f"{model_name:<25} {avg_latency:<20.2f} {avg_sources:<20.2f}")


def main():
    """
    Main function to compare embedding models.
    """
    print("\n" + "="*80)
    print("BONUS 2: Embedding Model Comparison")
    print("="*80)
    
    try:
        # Load documents and create chunks
        print("\nLoading documents...")
        documents = load_documents()
        chunks = split_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        
        # Initialize comparison
        comparator = EmbeddingComparison(documents, chunks)
        
        # Set up all models
        comparator.setup_all_models()
        
        if len(comparator.vector_stores) == 0:
            print("\n✗ No embedding models could be set up. Please check dependencies and API keys.")
            return
        
        # Compare models
        comparator.comprehensive_comparison(queries=TEST_QUERIES)
        
        print(f"\n{'='*80}")
        print("SUCCESS: Embedding Model Comparison Complete!")
        print(f"{'='*80}")
        print("\nModels Tested:")
        for model_name in comparator.vector_stores.keys():
            print(f"  ✓ {model_name}")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

