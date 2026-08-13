from datetime import datetime, timezone
from pathlib import Path

import pytest

from voice_agent.profiles import ProfileRevokedError, ProfileStore, VoiceProfile
from voice_agent.storage import SQLiteProfileRepository


class AllowConsent:
    def verify(self, profile: VoiceProfile) -> bool:
        return profile.consent_id == "consent-1"


def make_profile(reference_audio: Path) -> VoiceProfile:
    return VoiceProfile(
        profile_id="profile-1",
        consent_id="consent-1",
        consented_at=datetime(2026, 8, 13, tzinfo=timezone.utc),
        language="en",
        reference_audio_path=str(reference_audio),
    )


def test_registered_profile_survives_a_new_store_instance(tmp_path: Path) -> None:
    reference_audio = tmp_path / "reference.wav"
    reference_audio.write_bytes(b"reference-audio")
    database_path = tmp_path / "data" / "profiles.sqlite3"

    first_repository = SQLiteProfileRepository(str(database_path))
    first_store = ProfileStore(AllowConsent(), first_repository)
    profile = make_profile(reference_audio)
    first_store.register(profile)
    first_repository.close()

    second_repository = SQLiteProfileRepository(str(database_path))
    second_store = ProfileStore(AllowConsent(), second_repository)

    assert second_store.get(profile.profile_id) == profile
    assert second_store.assert_usable(profile.profile_id, "en").profile == profile
    second_repository.close()


def test_revoked_profile_remains_unusable_after_restart(tmp_path: Path) -> None:
    reference_audio = tmp_path / "reference.wav"
    reference_audio.write_bytes(b"reference-audio")
    database_path = tmp_path / "profiles.sqlite3"
    profile = make_profile(reference_audio)

    repository = SQLiteProfileRepository(str(database_path))
    store = ProfileStore(AllowConsent(), repository)
    store.register(profile)
    repository.revoke(profile.profile_id)
    repository.close()

    restarted_repository = SQLiteProfileRepository(str(database_path))
    restarted_store = ProfileStore(AllowConsent(), restarted_repository)

    with pytest.raises(ProfileRevokedError):
        restarted_store.assert_usable(profile.profile_id, "en")
    restarted_repository.close()


def test_repository_does_not_copy_raw_audio_bytes_into_sqlite(tmp_path: Path) -> None:
    reference_audio = tmp_path / "reference.wav"
    raw_audio = b"unique-raw-audio-payload-must-not-be-persisted"
    reference_audio.write_bytes(raw_audio)
    database_path = tmp_path / "profiles.sqlite3"

    repository = SQLiteProfileRepository(str(database_path))
    ProfileStore(AllowConsent(), repository).register(make_profile(reference_audio))
    repository.close()

    assert raw_audio not in database_path.read_bytes()
