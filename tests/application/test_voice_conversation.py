from threading import Event, Thread

import numpy as np
import pytest

from components.speech_text_normalizer import SpeechTextNormalizer
from components.voice_conversation import VoiceConversation
from ports import Message


class FakeDetector:
    def __init__(self):
        self.recordings = 0

    def record_turn(self):
        self.recordings += 1
        return np.array([0.1], dtype="float32")


class FakeActivation:
    def __init__(self):
        self.activations = 0

    def wait_for_activation(self):
        self.activations += 1


class SilentDetector:
    def record_turn(self):
        return None


class FakeRecognizer:
    def transcribe(self, _audio, _language=None):
        return "hello"


class FakeModel:
    def reply(self, messages):
        assert messages[-1] == Message("user", "hello")
        return "Hi there"


class FixedResponseModel:
    def __init__(self, response):
        self.response = response

    def reply(self, _messages):
        return self.response


class FakeSynthesizer:
    def __init__(self):
        self.spoken = []

    def speak(self, text):
        self.spoken.append(text)


def make_conversation(detector=None, recognizer=None, model=None, synthesizer=None, activation=None):
    return VoiceConversation(
        activation or FakeActivation(),
        detector or FakeDetector(),
        recognizer or FakeRecognizer(),
        model or FakeModel(),
        synthesizer or FakeSynthesizer(),
        SpeechTextNormalizer(),
    )


def test_turn_is_transcribed_replied_to_and_spoken():
    speaker = FakeSynthesizer()
    conversation = make_conversation(synthesizer=speaker)

    response = conversation.process_turn()

    assert response == "Hi there"
    assert speaker.spoken == ["Hi there"]
    assert conversation.last_message == Message("assistant", "Hi there")


def test_empty_audio_does_not_call_model():
    speaker = FakeSynthesizer()
    conversation = make_conversation(synthesizer=speaker)

    assert conversation.process_turn(np.array([], dtype="float32")) is None
    assert speaker.spoken == []


def test_last_message_reflects_most_recent_entry():
    conversation = make_conversation()

    assert conversation.last_message == Message("system", "You are Vaani, a friendly local voice assistant.")

    conversation.process_turn()

    assert conversation.last_message == conversation.messages[-1]


def test_multi_sample_ndarray_audio_does_not_raise():
    # Regression test: real VAD output is a multi-element np.ndarray, whose truthiness
    # is ambiguous. process_turn must check emptiness without relying on `if not audio`.
    speaker = FakeSynthesizer()
    conversation = make_conversation(synthesizer=speaker)
    audio = np.array([0.1, 0.2, 0.3], dtype="float32")

    response = conversation.process_turn(audio)

    assert response == "Hi there"
    assert speaker.spoken == ["Hi there"]


def test_empty_ndarray_audio_does_not_call_model():
    speaker = FakeSynthesizer()
    conversation = make_conversation(synthesizer=speaker)

    assert conversation.process_turn(np.array([], dtype="float32")) is None
    assert speaker.spoken == []


def test_detector_returning_none_does_not_call_model():
    # Regression test: when no audio argument is given, process_turn falls back to
    # detector.record_turn(), which returns None on silence. Must not crash on len(None).
    speaker = FakeSynthesizer()
    conversation = make_conversation(detector=SilentDetector(), synthesizer=speaker)

    assert conversation.process_turn() is None
    assert speaker.spoken == []


@pytest.mark.parametrize(
    ("model_response", "spoken_text"),
    [
        ("**Bold** and *italic* text.", "Bold and italic text."),
        ("# Heading", "Heading"),
        ("Read [the Vaani docs](https://example.com/vaani).", "Read the Vaani docs."),
        ("- first item\n* second item\n1. third item", "first item\nsecond item\nthird item"),
        (
            "Run `vaani --help`.\n```python\nprint('hello')\n```",
            "Run vaani --help.\nprint('hello')",
        ),
        ("Six times seven is 6 * 7 = 42.", "Six times seven is 6 * 7 = 42."),
        ("Plain text stays unchanged.", "Plain text stays unchanged."),
        ("**Hi there!** \U0001f44b", "Hi there!"),
        ("\U0001f44b\U0001f3fd Hello!", "Hello!"),
        ("Hello\U0001f44b!", "Hello!"),
        ("Family: \U0001f468\u200d\U0001f469\u200d\U0001f467", "Family:"),
        ("Great! \u2764\ufe0f \U0001f1ee\U0001f1f3 1\ufe0f\u20e3", "Great!"),
        ("Use `hello\U0001f44b`.", "Use hello."),
        ("Price: $5. Math: 6 * 7 = 42. Caf\u00e9.", "Price: $5. Math: 6 * 7 = 42. Caf\u00e9."),
    ],
)
def test_markdown_is_normalized_only_for_speech(model_response, spoken_text):
    speaker = FakeSynthesizer()
    conversation = make_conversation(model=FixedResponseModel(model_response), synthesizer=speaker)

    response = conversation.process_turn()

    assert speaker.spoken == [spoken_text]
    assert response == model_response
    assert conversation.last_message == Message("assistant", model_response)


def test_waits_for_activation_before_recording_a_microphone_turn():
    activation_entered = Event()
    release = Event()
    capture_started = Event()

    class BlockingActivation:
        def wait_for_activation(self):
            activation_entered.set()
            assert release.wait(5)

    class SignalingDetector(FakeDetector):
        def record_turn(self):
            capture_started.set()
            return super().record_turn()

    activation = BlockingActivation()
    detector = SignalingDetector()
    conversation = make_conversation(activation=activation, detector=detector)
    thread = Thread(target=conversation.process_turn)
    thread.start()
    try:
        assert activation_entered.wait(5)
        assert not capture_started.is_set()
    finally:
        release.set()
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert capture_started.is_set()
    assert detector.recordings == 1


def test_supplied_audio_bypasses_activation_and_microphone_capture():
    activation = FakeActivation()
    detector = FakeDetector()
    conversation = make_conversation(activation=activation, detector=detector)

    conversation.process_turn(np.array([0.1], dtype="float32"))

    assert activation.activations == 0
    assert detector.recordings == 0
