from unittest.mock import MagicMock

from adapters.activation import ImmediateActivation
from vaani.__main__ import main


def test_cli_wires_immediate_activation(monkeypatch):
    import vaani.__main__ as cli

    detector = MagicMock()
    conversation = MagicMock()
    conversation.return_value.process_turn.side_effect = KeyboardInterrupt
    monkeypatch.setattr(cli, "SileroTurnDetector", detector)
    monkeypatch.setattr(cli, "VoiceConversation", conversation)
    monkeypatch.setattr(cli, "FasterWhisperRecognizer", MagicMock())
    monkeypatch.setattr(cli, "OllamaModel", MagicMock())
    monkeypatch.setattr(cli, "PiperSynthesizer", MagicMock())
    monkeypatch.setattr("sys.argv", ["vaani", "--voice", "voice.onnx"])

    main()

    activation, actual_detector = conversation.call_args.args[:2]
    assert isinstance(activation, ImmediateActivation)
    assert activation.wait_for_activation() is None
    assert actual_detector is detector.return_value
    detector.assert_called_once_with()
