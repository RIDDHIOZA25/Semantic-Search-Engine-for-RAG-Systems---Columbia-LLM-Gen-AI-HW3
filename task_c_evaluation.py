"""
Task C: Evaluate Semantic Search Implementation

This module implements comprehensive evaluation:
1. Testing on provided dataset (10 diverse documents covering multiple topics:
   technology, science, sports, politics, history)
2. Comparing retrieval quality between chunk sizes (500, 1000, 1500 characters)
3. Measuring search latency for k=1, 5, 10
4. Analyzing diversity effects (similarity vs MMR)
5. Demonstrating MMR better coverage scenarios
6. Visualizing embedding similarities using dimensionality reduction (PCA/t-SNE)

Usage:
    python task_c_evaluation.py

Prerequisites:
    - Run task_a_document_processing.py first to create base vector store
    - Documents should be in ./data/ directory
    - Requires: matplotlib, scikit-learn, numpy (see requirements_task_c.txt)
"""

import os
import time
import shutil
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Import Task A functions
from task_a_document_processing import (
    load_documents,
    split_documents,
    initialize_embeddings,
    create_vector_store,
    DATA_DIR,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL_NAME
)

# Import Task B functions
from task_b_semantic_search import (
    similarity_search,
    similarity_search_with_score,
    mmr_search,
    TEST_QUERIES,
    MMR_LAMBDA
)

from langchain_community.vectorstores import Chroma
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings


# Configuration
CHUNK_SIZES = [500, 1000, 1500]  # Chunk sizes to compare
K_VALUES = [1, 5, 10]  # k values for latency measurement
PERSIST_DIR_BASE = "./chroma_rag_demo"
# CHUNK_OVERLAP is imported from task_a_document_processing


def create_vector_stores_for_chunk_sizes(
    chunk_sizes: List[int] = CHUNK_SIZES,
    chunk_overlap: int = CHUNK_OVERLAP
) -> Dict[int, Chroma]:
    """
    Create vector stores with different chunk sizes for comparison.
    
    Args:
        chunk_sizes: List of chunk sizes to test
        chunk_overlap: Overlap between chunks
        
    Returns:
        Dictionary mapping chunk_size -> Chroma vector store
    """
    print(f"\n{'='*80}")
    print("Creating Vector Stores with Different Chunk Sizes")
    print(f"{'='*80}")
    
    # Load documents once
    docs = load_documents()
    embeddings = initialize_embeddings()
    
    vector_stores = {}
    
    for chunk_size in chunk_sizes:
        print(f"\n{'─'*80}")
        print(f"Chunk Size: {chunk_size} characters")
        print(f"{'─'*80}")
        
        # Create unique persist directory for each chunk size
        persist_dir = f"{PERSIST_DIR_BASE}_chunk{chunk_size}"
        
        # Clean up if exists
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)
        
        # Split with this chunk size
        chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        # Create vector store
        vector_store = create_vector_store(
            chunks,
            embeddings,
            persist_directory=persist_dir,
            collection_name="rag_demo"
        )
        
        vector_stores[chunk_size] = vector_store
        print(f"✓ Vector store created for chunk_size={chunk_size}")
    
    return vector_stores


def compare_chunk_sizes(
    vector_stores: Dict[int, Chroma],
    query: str = TEST_QUERIES[0],
    k: int = 5
) -> Dict[int, Dict]:
    """
    Compare retrieval quality across different chunk sizes.
    
    Args:
        vector_stores: Dictionary of chunk_size -> vector_store
        query: Test query
        k: Number of results to retrieve
        
    Returns:
        Dictionary with comparison metrics
    """
    print(f"\n{'='*80}")
    print(f"Chunk Size Comparison for Query: '{query}'")
    print(f"{'='*80}")
    
    results = {}
    
    print(f"\n{'Chunk Size':<15} {'Total Chunks':<15} {'Unique Sources':<15} {'Avg Score':<15}")
    print("-" * 80)
    
    for chunk_size, vector_store in vector_stores.items():
        # Perform similarity search
        sim_results = similarity_search_with_score(vector_store, query, k=k)
        
        # Calculate metrics
        total_chunks = vector_store._collection.count()
        sources = {os.path.basename(doc.metadata.get('source', '')) for doc, _ in sim_results}
        avg_score = np.mean([score for _, score in sim_results])
        
        results[chunk_size] = {
            'total_chunks': total_chunks,
            'unique_sources': len(sources),
            'avg_score': avg_score,
            'sources': sources
        }
        
        print(f"{chunk_size:<15} {total_chunks:<15} {len(sources):<15} {avg_score:<15.4f}")
    
    # Analysis
    print(f"\n{'─'*80}")
    print("Analysis:")
    print(f"{'─'*80}")
    
    # Find which chunk size gives best diversity
    best_diversity = max(results.items(), key=lambda x: x[1]['unique_sources'])
    print(f"Best diversity ({best_diversity[1]['unique_sources']} unique sources): "
          f"chunk_size={best_diversity[0]}")
    
    # Find which chunk size gives best relevance (lowest avg score)
    best_relevance = min(results.items(), key=lambda x: x[1]['avg_score'])
    print(f"Best relevance (avg_score={best_relevance[1]['avg_score']:.4f}): "
          f"chunk_size={best_relevance[0]}")
    
    return results


def measure_comprehensive_latency(
    vector_store: Chroma,
    query: str,
    k_values: List[int] = K_VALUES,
    num_runs: int = 5
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """
    Comprehensive latency measurement for all search methods.
    
    Args:
        vector_store: Chroma vector store instance
        query: Test query
        k_values: List of k values to test
        num_runs: Number of runs to average
        
    Returns:
        Nested dictionary with latency results
    """
    print(f"\n{'='*80}")
    print(f"Comprehensive Latency Measurement")
    print(f"{'='*80}")
    print(f"Query: '{query}'")
    print(f"Runs per measurement: {num_runs}")
    
    results = {
        "similarity": {},
        "similarity_with_scores": {},
        "mmr": {}
    }
    
    for k in k_values:
        # Similarity Search
        times = []
        for _ in range(num_runs):
            start = time.time()
            similarity_search(vector_store, query, k=k)
            times.append((time.time() - start) * 1000)  # Convert to ms
        results["similarity"][k] = {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times)
        }
        
        # Similarity with Scores
        times = []
        for _ in range(num_runs):
            start = time.time()
            similarity_search_with_score(vector_store, query, k=k)
            times.append((time.time() - start) * 1000)
        results["similarity_with_scores"][k] = {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times)
        }
        
        # MMR Search
        times = []
        for _ in range(num_runs):
            start = time.time()
            mmr_search(vector_store, query, k=k)
            times.append((time.time() - start) * 1000)
        results["mmr"][k] = {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times)
        }
    
    # Display results
    print(f"\n{'Method':<25} {'k=1 (ms)':<15} {'k=5 (ms)':<15} {'k=10 (ms)':<15}")
    print("-" * 80)
    
    for method, method_results in results.items():
        row = f"{method.replace('_', ' ').title():<25}"
        for k in k_values:
            mean = method_results[k]['mean']
            std = method_results[k]['std']
            row += f"{mean:.2f}±{std:.2f}".ljust(15)
        print(row)
    
    return results


def analyze_diversity_effects(
    vector_store: Chroma,
    queries: List[str] = TEST_QUERIES,
    k: int = 5
) -> Dict:
    """
    Comprehensive analysis of diversity effects between similarity and MMR search.
    
    Args:
        vector_store: Chroma vector store instance
        queries: List of test queries
        k: Number of results
        
    Returns:
        Dictionary with diversity metrics
    """
    print(f"\n{'='*80}")
    print("Diversity Analysis: Similarity vs MMR Search")
    print(f"{'='*80}")
    
    diversity_results = {
        'similarity': {'unique_sources_total': set(), 'redundancy_count': 0},
        'mmr': {'unique_sources_total': set(), 'redundancy_count': 0}
    }
    
    all_results = {}
    
    for query in queries:
        print(f"\n{'─'*80}")
        print(f"Query: '{query}'")
        print(f"{'─'*80}")
        
        # Similarity search
        sim_results = similarity_search(vector_store, query, k=k)
        sim_sources = defaultdict(int)
        for doc in sim_results:
            source = os.path.basename(doc.metadata.get('source', 'unknown'))
            sim_sources[source] += 1
            diversity_results['similarity']['unique_sources_total'].add(source)
        
        # MMR search
        mmr_results = mmr_search(vector_store, query, k=k, lambda_param=MMR_LAMBDA)
        mmr_sources = defaultdict(int)
        for doc in mmr_results:
            source = os.path.basename(doc.metadata.get('source', 'unknown'))
            mmr_sources[source] += 1
            diversity_results['mmr']['unique_sources_total'].add(source)
        
        # Count redundancy (sources appearing multiple times)
        sim_redundancy = sum(1 for count in sim_sources.values() if count > 1)
        mmr_redundancy = sum(1 for count in mmr_sources.values() if count > 1)
        
        diversity_results['similarity']['redundancy_count'] += sim_redundancy
        diversity_results['mmr']['redundancy_count'] += mmr_redundancy
        
        all_results[query] = {
            'similarity': {'sources': sim_sources, 'unique_count': len(sim_sources)},
            'mmr': {'sources': mmr_sources, 'unique_count': len(mmr_sources)}
        }
        
        print(f"Similarity Search: {len(sim_sources)} unique sources, {sim_redundancy} redundant")
        print(f"MMR Search:        {len(mmr_sources)} unique sources, {mmr_redundancy} redundant")
        print(f"Improvement:       {len(mmr_sources) - len(sim_sources)} more unique sources with MMR")
    
    # Summary
    print(f"\n{'='*80}")
    print("Diversity Summary Across All Queries")
    print(f"{'='*80}")
    print(f"Similarity Search:")
    print(f"  Total unique sources: {len(diversity_results['similarity']['unique_sources_total'])}")
    print(f"  Total redundancy cases: {diversity_results['similarity']['redundancy_count']}")
    print(f"\nMMR Search:")
    print(f"  Total unique sources: {len(diversity_results['mmr']['unique_sources_total'])}")
    print(f"  Total redundancy cases: {diversity_results['mmr']['redundancy_count']}")
    
    improvement = len(diversity_results['mmr']['unique_sources_total']) - \
                  len(diversity_results['similarity']['unique_sources_total'])
    redundancy_reduction = diversity_results['similarity']['redundancy_count'] - \
                           diversity_results['mmr']['redundancy_count']
    
    print(f"\n✓ MMR provides {improvement} more unique sources overall")
    print(f"✓ MMR reduces redundancy by {redundancy_reduction} cases")
    
    return all_results, diversity_results


def demonstrate_mmr_coverage_scenarios(
    vector_store: Chroma,
    queries: List[str] = TEST_QUERIES,
    k: int = 5
):
    """
    Demonstrate specific scenarios where MMR provides better coverage.
    
    Args:
        vector_store: Chroma vector store instance
        queries: Test queries
        k: Number of results
    """
    print(f"\n{'='*80}")
    print("MMR Coverage Scenarios Demonstration")
    print(f"{'='*80}")
    
    best_scenarios = []
    
    for query in queries:
        sim_results = similarity_search(vector_store, query, k=k)
        mmr_results = mmr_search(vector_store, query, k=k, lambda_param=MMR_LAMBDA)
        
        sim_sources = {os.path.basename(d.metadata.get('source', '')) for d in sim_results}
        mmr_sources = {os.path.basename(d.metadata.get('source', '')) for d in mmr_results}
        
        mmr_only = mmr_sources - sim_sources
        improvement = len(mmr_sources) - len(sim_sources)
        
        if improvement > 0:
            best_scenarios.append({
                'query': query,
                'improvement': improvement,
                'mmr_only_sources': mmr_only,
                'sim_sources': sim_sources,
                'mmr_sources': mmr_sources
            })
    
    # Sort by improvement
    best_scenarios.sort(key=lambda x: x['improvement'], reverse=True)
    
    print(f"\nTop {min(3, len(best_scenarios))} scenarios where MMR provides better coverage:")
    print("-" * 80)
    
    for i, scenario in enumerate(best_scenarios[:3], 1):
        print(f"\nScenario {i}:")
        print(f"  Query: '{scenario['query']}'")
        print(f"  Improvement: {scenario['improvement']} more unique sources")
        print(f"  Similarity sources ({len(scenario['sim_sources'])}): {scenario['sim_sources']}")
        print(f"  MMR sources ({len(scenario['mmr_sources'])}): {scenario['mmr_sources']}")
        if scenario['mmr_only_sources']:
            print(f"  Sources only found by MMR: {scenario['mmr_only_sources']}")
    
    return best_scenarios


def visualize_embeddings(
    vector_store: Chroma,
    embeddings_model: HuggingFaceEmbeddings,
    sample_size: Optional[int] = None,
    method: str = 'pca'
):
    """
    Create visualization of embedding similarities using dimensionality reduction.
    
    Args:
        vector_store: Chroma vector store instance
        embeddings_model: Embedding model instance
        sample_size: Number of embeddings to visualize (None = all)
        method: 'pca' or 'tsne' for dimensionality reduction
    """
    print(f"\n{'='*80}")
    print(f"Embedding Visualization using {method.upper()}")
    print(f"{'='*80}")
    
    # Get embeddings and metadata
    all_data = vector_store._collection.get(
        include=["metadatas", "embeddings"],
        limit=sample_size
    )
    
    embedding_vectors = np.array(all_data['embeddings'])
    metadatas = all_data['metadatas']
    
    print(f"Total embeddings: {len(embedding_vectors)}")
    print(f"Embedding dimension: {embedding_vectors.shape[1]}")
    
    # Perform dimensionality reduction
    print(f"\nPerforming {method.upper()} dimensionality reduction...")
    
    if method.lower() == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        reduced = reducer.fit_transform(embedding_vectors)
        explained_var = sum(reducer.explained_variance_ratio_)
        print(f"Explained variance: {explained_var*100:.2f}%")
    elif method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embedding_vectors)-1))
        reduced = reducer.fit_transform(embedding_vectors)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'pca' or 'tsne'")
    
    # Extract source names for coloring
    source_names = [os.path.basename(meta.get('source', 'unknown')) for meta in metadatas]
    unique_sources = list(set(source_names))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_sources)))
    source_to_color = dict(zip(unique_sources, colors))
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(14, 10))
    
    for source in unique_sources:
        indices = [i for i, s in enumerate(source_names) if s == source]
        points = reduced[indices]
        ax.scatter(
            points[:, 0],
            points[:, 1],
            label=source.replace('.txt', ''),
            color=source_to_color[source],
            alpha=0.6,
            s=50
        )
    
    ax.set_xlabel(f'{method.upper()} Component 1', fontsize=12)
    ax.set_ylabel(f'{method.upper()} Component 2', fontsize=12)
    ax.set_title(f'Embedding Similarities Visualization ({method.upper()})', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_file = f'embedding_visualization_{method}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_file}")
    
    # Also create a second plot showing query embeddings
    if len(TEST_QUERIES) > 0:
        print(f"\nAdding query embeddings to visualization...")
        
        # Embed queries
        query_embeddings = []
        for query in TEST_QUERIES:
            emb = embeddings_model.embed_query(query)
            query_embeddings.append(emb)
        
        query_embeddings = np.array(query_embeddings)
        
        # Transform queries
        if method.lower() == 'pca':
            query_reduced = reducer.transform(query_embeddings)
        else:  # t-SNE needs to refit with queries included
            print("  Note: t-SNE requires refitting with queries included")
            combined = np.vstack([embedding_vectors, query_embeddings])
            reducer_combined = TSNE(n_components=2, random_state=42, 
                                   perplexity=min(30, len(combined)-1))
            combined_reduced = reducer_combined.fit_transform(combined)
            query_reduced = combined_reduced[-len(TEST_QUERIES):]
            reduced = combined_reduced[:-len(TEST_QUERIES)]
        
        # Create combined visualization
        fig2, ax2 = plt.subplots(figsize=(14, 10))
        
        # Plot document embeddings
        for source in unique_sources:
            indices = [i for i, s in enumerate(source_names) if s == source]
            points = reduced[indices]
            ax2.scatter(
                points[:, 0],
                points[:, 1],
                label=source.replace('.txt', ''),
                color=source_to_color[source],
                alpha=0.4,
                s=30
            )
        
        # Plot query embeddings
        for i, (query, point) in enumerate(zip(TEST_QUERIES, query_reduced)):
            ax2.scatter(
                point[0],
                point[1],
                marker='*',
                s=500,
                label=f"Query: {query[:30]}...",
                edgecolors='black',
                linewidths=2,
                alpha=0.9
            )
        
        ax2.set_xlabel(f'{method.upper()} Component 1', fontsize=12)
        ax2.set_ylabel(f'{method.upper()} Component 2', fontsize=12)
        ax2.set_title(f'Embeddings with Queries ({method.upper()})', fontsize=14, fontweight='bold')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file2 = f'embedding_visualization_{method}_with_queries.png'
        plt.savefig(output_file2, dpi=300, bbox_inches='tight')
        print(f"✓ Query visualization saved to: {output_file2}")
    
    plt.close('all')


def main():
    """
    Main function to execute Task C: Comprehensive Evaluation
    """
    print("\n" + "="*80)
    print("TASK C: Comprehensive Evaluation of Semantic Search")
    print("="*80)
    
    try:
        # Initialize embeddings
        embeddings = initialize_embeddings()
        
        # 1. Create vector stores with different chunk sizes
        vector_stores = create_vector_stores_for_chunk_sizes(CHUNK_SIZES)
        
        # 2. Compare chunk sizes
        print(f"\n{'='*80}")
        print("EVALUATION 1: Chunk Size Comparison")
        print(f"{'='*80}")
        chunk_comparison = compare_chunk_sizes(vector_stores, query=TEST_QUERIES[0])
        
        # Use the default chunk size (1000) for remaining evaluations
        default_vector_store = vector_stores[1000]
        
        # 3. Measure comprehensive latency
        print(f"\n{'='*80}")
        print("EVALUATION 2: Latency Measurement")
        print(f"{'='*80}")
        latency_results = measure_comprehensive_latency(
            default_vector_store,
            TEST_QUERIES[0],
            k_values=K_VALUES
        )
        
        # 4. Analyze diversity effects
        print(f"\n{'='*80}")
        print("EVALUATION 3: Diversity Analysis")
        print(f"{'='*80}")
        diversity_results, diversity_summary = analyze_diversity_effects(
            default_vector_store,
            queries=TEST_QUERIES
        )
        
        # 5. Demonstrate MMR coverage scenarios
        print(f"\n{'='*80}")
        print("EVALUATION 4: MMR Coverage Scenarios")
        print(f"{'='*80}")
        mmr_scenarios = demonstrate_mmr_coverage_scenarios(
            default_vector_store,
            queries=TEST_QUERIES
        )
        
        # 6. Create visualizations
        print(f"\n{'='*80}")
        print("EVALUATION 5: Embedding Visualizations")
        print(f"{'='*80}")
        
        # PCA visualization
        visualize_embeddings(default_vector_store, embeddings, method='pca', sample_size=100)
        
        # t-SNE visualization (on smaller sample due to computational cost)
        print("\nNote: t-SNE is computationally expensive, using sample of 50 embeddings")
        visualize_embeddings(default_vector_store, embeddings, method='tsne', sample_size=50)
        
        # Final summary
        print(f"\n{'='*80}")
        print("SUCCESS: Task C Evaluation Complete!")
        print(f"{'='*80}")
        print("\nSummary of Evaluations:")
        print("  ✓ Tested on dataset (10 diverse documents)")
        print("  ✓ Compared chunk sizes: 500, 1000, 1500")
        print("  ✓ Measured latency for k=1, 5, 10")
        print("  ✓ Analyzed diversity effects (similarity vs MMR)")
        print("  ✓ Demonstrated MMR coverage scenarios")
        print("  ✓ Created embedding visualizations (PCA and t-SNE)")
        print(f"\nVisualization files created:")
        print("  - embedding_visualization_pca.png")
        print("  - embedding_visualization_pca_with_queries.png")
        print("  - embedding_visualization_tsne.png")
        print("  - embedding_visualization_tsne_with_queries.png")
        
    except Exception as e:
        print(f"\n✗ Error in Task C: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

