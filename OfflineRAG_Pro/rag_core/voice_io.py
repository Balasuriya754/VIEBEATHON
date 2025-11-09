import os, tempfile, threading, queue, time, numpy as np
from typing import Optional, Callable, Tuple

try:
    import sounddevice as sd
    import soundfile as sf
    import whisper
    AUDIO_AVAILABLE = True
except:
    AUDIO_AVAILABLE = False
    print("⚠️ Missing audio deps — run: pip install sounddevice soundfile openai-whisper")

try:
    import pyttsx3
    TTS_AVAILABLE = True
    print("✅ Using pyttsx3 (offline TTS)")
except:
    TTS_AVAILABLE = False
    print("⚠️ No TTS engine found. Run: pip install pyttsx3")

# ==========================================================
# VOICE INPUT (Whisper)
# ==========================================================
class VoiceInput:
    def __init__(self, model_size="base", lang="en"):
        if not AUDIO_AVAILABLE:
            raise ImportError("Audio libs missing.")
        print(f"🎤 Loading Whisper model ({model_size})...")
        self.model = whisper.load_model(model_size)
        self.lang = lang
        print("✅ Whisper ready.")

    def transcribe(self, audio: np.ndarray) -> Tuple[str, dict]:
        tmp = tempfile.mktemp(suffix=".wav")
        sf.write(tmp, audio, 16000)
        res = self.model.transcribe(tmp, language=self.lang, fp16=False)
        try:
            os.remove(tmp)
        except:
            pass
        return res.get("text", "").strip(), res

# ==========================================================
# VOICE OUTPUT (pyttsx3)
# ==========================================================
class VoiceOutput:
    def __init__(self):
        if not TTS_AVAILABLE:
            raise ImportError("pyttsx3 missing.")
        self.q = queue.Queue()
        self.running = True
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        engine = pyttsx3.init()
        engine.setProperty("rate", 175)
        engine.setProperty("volume", 1.0)
        while self.running:
            try:
                text, stop_check = self.q.get(timeout=0.5)
                if stop_check and stop_check():
                    continue
                engine.say(text)
                engine.runAndWait()
                self.q.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"TTS error: {e}")

    def speak(self, text, stop_check: Optional[Callable[[], bool]] = None):
        if text.strip():
            self.q.put((text, stop_check))

    def stop(self):
        try:
            while not self.q.empty():
                self.q.get_nowait()
                self.q.task_done()
            print("🛑 Speech stopped")
        except:
            pass

# ==========================================================
# VOICE ASSISTANT
# ==========================================================
class VoiceAssistant:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *a, **k):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, whisper_model="base"):
        if hasattr(self, "_initialized") and self._initialized:
            return
        self.voice_input = VoiceInput(whisper_model)
        self.voice_output = VoiceOutput()
        self._initialized = True
        print("✅ Voice Assistant initialized (offline).")

    def transcribe(self, audio):
        return self.voice_input.transcribe(audio)

    def speak(self, text, stop_check=None):
        self.voice_output.speak(text, stop_check=stop_check)

    def stop_speaking(self):
        self.voice_output.stop()
