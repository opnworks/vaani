from __future__ import annotations

import numpy as np

from components.speech_text_normalizer import SpeechTextNormalizer
from ports import ActivationProvider, ConversationModel, Message, SpeechRecognizer, SpeechSynthesizer, VoiceActivityDetector

DEFAULT_SYSTEM_INSTRUCTION = "You are Vaani, a friendly local voice assistant."


class VoiceConversation:
    """Drives one turn of listen -> transcribe -> reply -> speak."""

    def __init__(self, activation_provider: ActivationProvider, detector: VoiceActivityDetector, recognizer: SpeechRecognizer,
                 model: ConversationModel, synthesizer: SpeechSynthesizer,
                 speech_text_normalizer: SpeechTextNormalizer,
                 language: str | None = None) -> None:
        self._activation_provider = activation_provider
        self._detector = detector
        self._recognizer = recognizer
        self._model = model
        self._synthesizer = synthesizer
        self._language = language
        self._speech_text_normalizer = speech_text_normalizer
        self._messages = [Message("system", DEFAULT_SYSTEM_INSTRUCTION)]


    def clear_context(self) -> None:
        self._messages = [Message("system", DEFAULT_SYSTEM_INSTRUCTION)]

    @property
    def messages(self) -> tuple[Message, ...]:
        return tuple(self._messages)

    @property
    def last_message(self) -> Message:
        return self._messages[-1]

    def process_turn(self, audio: np.ndarray | None = None) -> str | None:
        if audio is None:
            self._activation_provider.wait_for_activation()
            audio = self._detector.record_turn()
        if audio is None or len(audio) == 0:
            return None
        text = self._recognizer.transcribe(audio, self._language)
        if not text:
            return None
        if text.strip().casefold().rstrip(".!?").strip() == "clear conversation":
            self.clear_context()
            confirmation = "Conversation cleared. Starting fresh."
            self._synthesizer.speak(confirmation)
            return confirmation

        self._messages.append(Message("user", text))
        response = self._model.reply(self._messages)
        normalized_response = self._speech_text_normalizer.normalize(response)
        self._messages.append(Message("assistant", response))
        self._synthesizer.speak(normalized_response)
        return response
