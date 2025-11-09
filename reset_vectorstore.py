# reset_vectorstore.py
from OfflineRAG_Pro.rag_core.retriever import VectorStore

if __name__ == "__main__":
    vs = VectorStore("rag_storage/chroma")
    vs.reset()
