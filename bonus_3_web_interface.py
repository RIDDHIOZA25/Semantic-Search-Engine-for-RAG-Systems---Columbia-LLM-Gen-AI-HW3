"""
Bonus 3: Simple Web Interface for Semantic Search

This module creates a simple Streamlit web interface to demonstrate the search engine.

Usage:
    streamlit run bonus_3_web_interface.py

Prerequisites:
    - Run task_a_document_processing.py first
    - Install streamlit: pip install streamlit
"""

import os
import sys

try:
    import streamlit as st
except ImportError:
    print("ERROR: Streamlit not installed. Please run: pip install streamlit")
    sys.exit(1)

from langchain_community.vectorstores import Chroma
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from task_b_semantic_search import (
    similarity_search,
    similarity_search_with_score,
    mmr_search,
    PERSIST_DIR,
    COLLECTION,
    EMBEDDING_MODEL_NAME,
    MMR_LAMBDA
)


@st.cache_resource
def load_vector_store():
    """Load vector store (cached for Streamlit)."""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )
    
    if not os.path.exists(PERSIST_DIR):
        st.error(f"Vector store not found at {PERSIST_DIR}. Please run task_a_document_processing.py first.")
        st.stop()
    
    vector_store = Chroma(
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION,
        embedding_function=embeddings
    )
    
    return vector_store


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Semantic Search Engine",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Semantic Search Engine")
    st.markdown("**RAG System with Similarity Search and MMR**")
    
    # Load vector store
    vector_store = load_vector_store()
    total_docs = vector_store._collection.count()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Search Settings")
        
        search_method = st.radio(
            "Search Method",
            ["Similarity Search", "Similarity with Scores", "MMR Search"],
            index=0
        )
        
        k = st.slider("Number of Results (k)", min_value=1, max_value=20, value=5)
        
        if search_method == "MMR Search":
            lambda_param = st.slider("MMR λ Parameter", min_value=0.0, max_value=1.0, value=MMR_LAMBDA, step=0.1)
            st.caption(f"λ={lambda_param}: Higher = more relevant, Lower = more diverse")
        else:
            lambda_param = MMR_LAMBDA
        
        st.markdown("---")
        st.markdown(f"**Vector Store Info**")
        st.caption(f"Total documents: {total_docs}")
        st.caption(f"Collection: {COLLECTION}")
        
        st.markdown("---")
        st.markdown("**Sample Queries**")
        sample_queries = [
            "artificial intelligence applications",
            "climate change effects",
            "Olympic games history",
            "economic policy impacts"
        ]
        for sq in sample_queries:
            if st.button(sq, key=f"sample_{sq}", use_container_width=True):
                st.session_state.query = sq
    
    # Main area
    query = st.text_input(
        "Enter your search query:",
        value=st.session_state.get('query', ''),
        placeholder="e.g., 'artificial intelligence applications'"
    )
    
    if st.button("🔎 Search", type="primary", use_container_width=True):
        if not query:
            st.warning("Please enter a search query.")
        else:
            with st.spinner("Searching..."):
                try:
                    # Perform search based on method
                    if search_method == "Similarity Search":
                        results = similarity_search(vector_store, query, k=k)
                        display_results = [(doc, None) for doc in results]
                    elif search_method == "Similarity with Scores":
                        results = similarity_search_with_score(vector_store, query, k=k)
                        display_results = [(doc, score) for doc, score in results]
                    else:  # MMR Search
                        results = mmr_search(vector_store, query, k=k, lambda_param=lambda_param)
                        display_results = [(doc, None) for doc in results]
                    
                    # Display results
                    st.success(f"Found {len(display_results)} results")
                    
                    for i, (doc, score) in enumerate(display_results, 1):
                        with st.expander(f"Result {i}: {os.path.basename(doc.metadata.get('source', 'unknown'))}"):
                            if score is not None:
                                st.metric("Similarity Score", f"{1.0/(1.0+score):.4f}", 
                                         delta=None if i == 1 else f"{(1.0/(1.0+display_results[i-2][1])) - (1.0/(1.0+score)):.4f}")
                            else:
                                st.caption(f"Source: {doc.metadata.get('source', 'unknown')}")
                            
                            if 'start_index' in doc.metadata:
                                st.caption(f"Start Index: {doc.metadata['start_index']}")
                            
                            st.markdown("**Content:**")
                            st.write(doc.page_content)
                    
                    # Show source diversity
                    sources = [os.path.basename(doc.metadata.get('source', '')) for doc, _ in display_results]
                    unique_sources = set(sources)
                    
                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Results", len(display_results))
                    with col2:
                        st.metric("Unique Sources", len(unique_sources))
                    
                    if len(unique_sources) < len(sources):
                        st.info(f"⚠ Some sources appear multiple times: {Counter(sources).most_common(3)}")
                    
                except Exception as e:
                    st.error(f"Error performing search: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        <p>Semantic Search Engine | Built with LangChain & Chroma</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    # Check if running with streamlit
    import sys
    if 'streamlit' not in sys.modules:
        print("\n" + "="*80)
        print("STREAMLIT WEB INTERFACE")
        print("="*80)
        print("\nThis is a Streamlit web application.")
        print("To run it, use the following command:")
        print("\n  streamlit run bonus_3_web_interface.py")
        print("\nNOT: python bonus_3_web_interface.py")
        print("\n" + "="*80)
        sys.exit(1)
    main()

