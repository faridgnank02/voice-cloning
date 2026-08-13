import struct

import pytest

from voice_agent.audio import AudioFormat, AudioTurnAssembler, decode_browser_chunk


def pcm(samples):
    return struct.pack("<" + "h" * len(samples), *samples)


def test_decodes_mono_pcm_frames():
    frame = decode_browser_chunk(pcm([0, 1000, -1000]), "audio/pcm;rate=16000;channels=1")

    assert frame.sample_rate == 16000
    assert frame.channels == 1
    assert frame.data == pcm([0, 1000, -1000])


def test_rejects_unknown_content_type_and_oversized_frame():
    with pytest.raises(ValueError, match="unsupported audio format"):
        decode_browser_chunk(b"audio", "audio/webm")
    with pytest.raises(ValueError, match="audio frame limit"):
        decode_browser_chunk(b"x" * 100, "audio/pcm;rate=16000;channels=1", max_bytes=10)


def test_emits_speech_start_and_stop_after_silence():
    assembler = AudioTurnAssembler(AudioFormat(16000, 1, 2, "pcm_s16le"), silence_ms=20)

    assert [event.type for event in assembler.push(pcm([1000] * 160))] == ["speech_started"]
    assert assembler.push(pcm([0] * 160)) == []
    assert assembler.push(pcm([0] * 160))[0].type == "speech_stopped"
    assert assembler.flush() == pcm([1000] * 160 + [0] * 320)


def test_enforces_turn_byte_limit():
    assembler = AudioTurnAssembler(AudioFormat(16000, 1, 2, "pcm_s16le"), max_bytes=4)
    with pytest.raises(ValueError, match="turn audio limit"):
        assembler.push(pcm([1, 2, 3]))
