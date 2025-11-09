import time
import threading
import os
from pathlib import Path
from typing import List, Callable, Set
from datetime import datetime

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    print("⚠️ watchdog not installed. Run: pip install watchdog")


class WatcherHandler(FileSystemEventHandler):
    """
    Handles filesystem events with debouncing.
    Calls the given callback when files are created or modified.
    """

    def __init__(self, callback: Callable[[List[str]], None], watch_exts: List[str], debounce_sec: float = 1.5):
        self.callback = callback
        self.watch_exts = [e.lower() for e in watch_exts]
        self.debounce_sec = debounce_sec
        self._lock = threading.Lock()
        self._pending_files: Set[str] = set()
        self._timer = None
        self._processed_files: Set[str] = set()  # Track processed files to avoid duplicates

    def _schedule_callback(self):
        """Run callback after debounce delay."""

        def run():
            with self._lock:
                files = [f for f in self._pending_files if f not in self._processed_files]
                self._pending_files.clear()
            if files:
                self.callback(files)
                with self._lock:
                    self._processed_files.update(files)

        if self._timer:
            self._timer.cancel()
        self._timer = threading.Timer(self.debounce_sec, run)
        self._timer.start()

    def _interesting(self, path: str) -> bool:
        """Check if file extension is in watch list."""
        return Path(path).suffix.lower() in self.watch_exts

    def on_modified(self, event):
        if not event.is_directory and self._interesting(event.src_path):
            with self._lock:
                self._pending_files.add(event.src_path)
            self._schedule_callback()

    def on_created(self, event):
        if not event.is_directory and self._interesting(event.src_path):
            with self._lock:
                self._pending_files.add(event.src_path)
            self._schedule_callback()


class FolderWatcher:
    """
    Watches folders for file changes and automatically re-ingests updated documents.
    """

    def __init__(self, folders: List[str], on_files_changed: Callable[[List[str]], None],
                 exts: List[str] = None, debounce_sec: float = 2.0, log_dir: str = "./rag_storage/logs"):
        if not WATCHDOG_AVAILABLE:
            raise ImportError("watchdog library required. Install: pip install watchdog")

        self.folders = [Path(f) for f in folders]
        self.exts = exts or [".pdf", ".docx", ".txt", ".md", ".pptx", ".xlsx",
                             ".jpg", ".png", ".jpeg", ".mp3", ".wav", ".mp4", ".mov"]
        self.handler = WatcherHandler(on_files_changed, self.exts, debounce_sec)
        self.observer = Observer()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "watcher.log"
        self.running = False

    def _log(self, msg: str):
        """Thread-safe logging."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as e:
            print(f"⚠️ Logging error: {e}")

    def start(self):
        """Start watching the folders."""
        if self.running:
            self._log("⚠️ Watcher already running")
            return

        self.running = True
        scheduled_count = 0

        for folder in self.folders:
            if not folder.exists():
                self._log(f"⚠️ Folder not found (creating): {folder}")
                folder.mkdir(parents=True, exist_ok=True)

            self._log(f"👀 Watching folder: {folder}")
            self.observer.schedule(self.handler, str(folder), recursive=True)
            scheduled_count += 1

        if scheduled_count > 0:
            self.observer.start()
            self._log(f"✅ Watcher started monitoring {scheduled_count} folder(s)")
        else:
            self._log("⚠️ No folders to watch")
            self.running = False

    def stop(self):
        """Stop watching gracefully."""
        if not self.running:
            return

        self.running = False
        self.observer.stop()
        self.observer.join(timeout=5)
        self._log("🛑 Watcher stopped")

    def is_running(self) -> bool:
        """Check if watcher is active."""
        return self.running and self.observer.is_alive()