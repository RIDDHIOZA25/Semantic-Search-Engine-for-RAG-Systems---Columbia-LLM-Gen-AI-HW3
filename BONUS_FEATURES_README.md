# Bonus Features Implementation

This document describes the three bonus features implemented for Problem 2.

## Bonus 1: Hybrid Search (BM25 + Semantic) - +4 points

**File:** `bonus_1_hybrid_search.py`

### Features:
- Combines semantic search (vector similarity) with BM25 keyword-based scoring
- Configurable weights for semantic vs BM25 components
- Side-by-side comparison of all three methods (semantic, BM25, hybrid)

### Usage:
```bash
pip install rank-bm25
python bonus_1_hybrid_search.py
```

### How it works:
1. Performs semantic search using vector embeddings
2. Performs BM25 keyword matching on tokenized documents
3. Normalizes both scores to 0-1 range
4. Combines scores using weighted average: `hybrid_score = α × semantic + β × bm25`
5. Returns top-k results sorted by hybrid score

### Benefits:
- Semantic search captures meaning and context
- BM25 captures exact keyword matches
- Hybrid approach leverages both strengths

---

## Bonus 2: Embedding Model Comparison - +3 points

**File:** `bonus_2_embedding_comparison.py`

### Features:
- Compares multiple embedding models:
  - sentence-transformers/all-MiniLM-L6-v2 (default, small, fast)
  - sentence-transformers/all-mpnet-base-v2 (larger, higher quality)
  - OpenAI text-embedding-3-small (if API key available)
  - Cohere embed-english-v3.0 (if API key available)
- Measures latency and retrieval quality
- Side-by-side comparison of results

### Usage:
```bash
# For OpenAI (optional):
export OPENAI_API_KEY=your_key

# For Cohere (optional):
export COHERE_API_KEY=your_key

python bonus_2_embedding_comparison.py
```

### Metrics Compared:
- Average latency per query
- Number of unique sources retrieved
- Retrieval quality differences

---

## Bonus 3: Web Interface - +3 points

**File:** `bonus_3_web_interface.py`

### Features:
- Interactive Streamlit web interface
- Choose search method (Similarity, Similarity with Scores, MMR)
- Adjustable parameters (k, λ for MMR)
- Real-time search results
- Source diversity metrics
- Sample query buttons

### Usage:
```bash
pip install streamlit
streamlit run bonus_3_web_interface.py
```

### Interface Features:
- Search bar with query input
- Method selection (radio buttons)
- Sliders for k and λ parameters
- Expandable result cards
- Source diversity display
- Vector store information sidebar

---

## Installation

Install all bonus dependencies:
```bash
pip install -r requirements_bonus.txt
```

Or install individually:
- Bonus 1: `pip install rank-bm25`
- Bonus 2: No additional dependencies (optional: `pip install openai cohere`)
- Bonus 3: `pip install streamlit`

---

## Important Notes

1. **No Interference**: All bonus features use separate vector stores or work with existing ones without modifying Task A/B/C outputs.

2. **Prerequisites**: 
   - Run `task_a_document_processing.py` first to create the base vector store
   - Bonus 1 and 3 use the same vector store from Task A
   - Bonus 2 creates new vector stores for each model (doesn't affect Task A)

3. **Optional Features**:
   - OpenAI and Cohere embeddings require API keys
   - If not available, only local models (MiniLM, MPNet) will be tested

4. **File Organization**:
   - Bonus files are separate Python modules
   - They can be run independently
   - No modifications to Task A, B, or C files

---

## Testing the Bonuses

1. **Test Bonus 1** (Hybrid Search):
   ```bash
   python bonus_1_hybrid_search.py
   ```

2. **Test Bonus 2** (Embedding Comparison):
   ```bash
   python bonus_2_embedding_comparison.py
   ```

3. **Test Bonus 3** (Web Interface):
   ```bash
   streamlit run bonus_3_web_interface.py
   ```
   Then open http://localhost:8501 in your browser

---

## Expected Outputs

### Bonus 1:
- Comparison tables showing semantic, BM25, and hybrid results
- Source diversity metrics
- Score breakdowns

### Bonus 2:
- Model initialization messages
- Comparison tables with latency and quality metrics
- Per-query and summary statistics

### Bonus 3:
- Interactive web interface
- Real-time search results
- Visual metrics and statistics

