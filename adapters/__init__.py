"""Concrete implementations of the ports used by Vaani."""

from adapters.activation import ImmediateActivation
from adapters.conversation_model import OllamaModel
from adapters.recognizer import FasterWhisperRecognizer
from adapters.synthesizer import PiperSynthesizer
from adapters.silero_turn_detector import SileroTurnDetector

__all__ = ["ImmediateActivation", "OllamaModel", "FasterWhisperRecognizer", "PiperSynthesizer", "SileroTurnDetector"]
