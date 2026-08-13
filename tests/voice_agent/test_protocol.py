from voice_agent.protocol import event_json


def test_event_json_encodes_binary_audio():
    event = event_json("response_audio", "session-1", 2, audio=b"abc")

    assert event == {
        "type": "response_audio",
        "session_id": "session-1",
        "turn_id": 2,
        "audio": {"encoding": "base64", "data": "YWJj"},
    }
