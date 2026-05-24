import sys
from types import SimpleNamespace
from unittest.mock import patch

from speech_proto.audio_source import list_input_devices


class FakeSoundDevice:
    default = SimpleNamespace(device=[2, None])

    @staticmethod
    def query_devices():
        return [
            {"name": "Speaker", "max_input_channels": 0, "hostapi": 0, "default_samplerate": 48000},
            {"name": "USB Mic", "max_input_channels": 1, "hostapi": 0, "default_samplerate": 44100},
            {"name": "I2S Array", "max_input_channels": 2, "hostapi": 1, "default_samplerate": 16000},
        ]

    @staticmethod
    def query_hostapis():
        return [{"name": "CoreAudio"}, {"name": "ALSA"}]


def test_list_input_devices_filters_and_marks_default():
    with patch.dict(sys.modules, {"sounddevice": FakeSoundDevice}):
        devices = list_input_devices()

    assert [device.name for device in devices] == ["USB Mic", "I2S Array"]
    assert devices[0].host_api == "CoreAudio"
    assert devices[1].is_default is True

