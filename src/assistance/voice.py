"""Local Audio & Speech Output Assistance Module."""

import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)


class VoiceAssistance:
    """Non-blocking local text-to-speech audio dispatcher."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self._engine = None

        if self.enabled:
            try:
                import pyttsx3
                self._engine = pyttsx3.init()
            except Exception as e:
                logger.warning(f"Failed to initialize pyttsx3 TTS: {e}. Voice output disabled.")
                self.enabled = False

    def speak(self, text: str) -> None:
        """Speaks text message asynchronously on background thread."""
        if not self.enabled or not text:
            logger.info(f"[VOICE SPOKEN]: {text}")
            return

        def _worker():
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                logger.error(f"Error in TTS playback: {e}")

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
