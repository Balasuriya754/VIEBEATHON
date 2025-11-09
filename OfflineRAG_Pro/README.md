
# Offline RAG Pro (100% Offline, ChatGPT-like)

A modular, production-grade offline Retrieval-Augmented Generation system with a clean chat UI.

## Features
- **Offline only**: Uses local Ollama, ChromaDB, SentenceTransformers, Whisper, Tesseract.
- **Multi-modal ingestion**: PDF, DOCX, Images (OCR), Audio (Whisper).
- **Fast chat UI**: Claude/ChatGPT-like, streaming tokens, citations footer.
- **Persistent index**: Stored under `storage/`.

## Quick Start
```bash
# 1) Start a light local model
ollama pull llama3.2:1b
ollama serve

# 2) Install deps (Python 3.10+ recommended)
pip install -r requirements.txt

# 3) Launch
python app.py
# Open http://127.0.0.1:7860
```

## Notes
- Set a different model: `export OLLAMA_MODEL=mistral:7b-instruct`
- If Tesseract is not in PATH, install it and ensure `pytesseract` can find the binary.
- Whisper `tiny` is used for speed; change in `text_extractor.py` if needed.
