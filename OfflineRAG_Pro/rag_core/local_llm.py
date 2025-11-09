# ==========================================================
# local_llm.py — Offline Ollama Integration for BI AI
# ==========================================================
"""
Provides a unified generator interface for offline model inference via Ollama.
This connects BI AI to local LLMs like Llama 3, Mistral, Phi-3, etc.
"""

from typing import Generator
import requests
import os
import time
import json

# Default Ollama settings
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")

def generate(
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.2,
    stream: bool = True,
    retries: int = 2
) -> Generator[str, None, None]:
    """
    Stream or return complete responses from local Ollama.
    - Supports retries, streaming, and graceful sentence completion.
    """

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt.strip(),
        "stream": stream,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "stop": ["User:", "Assistant:"],
        },
    }

    for attempt in range(retries):
        try:
            if stream:
                with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=60) as r:
                    r.raise_for_status()
                    buffer = ""
                    for line in r.iter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line.decode("utf-8"))
                            token = data.get("response", "")
                            if token:
                                buffer += token
                                yield token
                            if data.get("done"):
                                break
                        except Exception:
                            continue

                    # End gracefully
                    if not buffer.endswith(('.', '!', '?')):
                        yield "."

                return

            else:
                # Non-streamed version
                r = requests.post(OLLAMA_URL, json=payload, timeout=60)
                r.raise_for_status()
                result = r.json().get("response", "").strip()
                if result:
                    if not result.endswith(('.', '!', '?')):
                        result += "."
                    yield result
                return

        except requests.exceptions.ConnectionError:
            if attempt < retries - 1:
                yield "⚠️ Connecting to Ollama... retrying.\n"
                time.sleep(3)
            else:
                yield "⚠️ Ollama not reachable. Run `ollama serve` and pull your model."
                return

        except Exception as e:
            yield f"⚠️ Local LLM error: {str(e)}"
            return
