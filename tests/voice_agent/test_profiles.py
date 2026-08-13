from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import voice_agent.profiles as profiles
from voice_agent.profiles import (
    ProfileError,
    ProfileLanguageMismatchError,
    ProfileNotFoundError,
    ProfileRevokedError,
    ProfileStore,
    ReferenceAudioNotFoundError,
    VoiceProfile,
)
from voice_agent.storage import SQLiteProfileRepository


class FakeConsentVerifier:
    def __init__(self, verified_consent_ids: set[str]) -> None:
        self.verified_consent_ids = verified_consent_ids
        self.checked_profiles: list[VoiceProfile] = []

    def verify(self, profile: VoiceProfile) -> bool:
        self.checked_profiles.append(profile)
        return profile.consent_id in self.verified_consent_ids


class UnavailableConsentVerifier:
    def verify(self, profile: VoiceProfile) -> bool:
        raise RuntimeError("consent service is unavailable")


class MutableConsentVerifier:
    def __init__(self, result: object = True) -> None:
        self.result = result
        self.checked_profiles: list[VoiceProfile] = []

    def verify(self, profile: VoiceProfile) -> object:
        self.checked_profiles.append(profile)
        return self.result


def make_store(
    verifier: FakeConsentVerifier | UnavailableConsentVerifier | None = None,
) -> ProfileStore:
    return ProfileStore(verifier or FakeConsentVerifier({"consent-1"}))


def make_profile(reference_audio_path: str, **overrides: object) -> VoiceProfile:
    values: dict[str, object] = {
        "profile_id": "profile-1",
        "consent_id": "consent-1",
        "consented_at": datetime(2026, 7, 18, tzinfo=timezone.utc),
        "language": "en",
        "reference_audio_path": reference_audio_path,
    }
    values.update(overrides)
    return VoiceProfile(**values)  # type: ignore[arg-type]


def test_registers_and_returns_a_usable_profile(tmp_path: Path) -> None:
    reference_audio = tmp_path / "reference.wav"
    reference_audio.write_bytes(b"audio")
    profile = make_profile(str(reference_audio))
    verifier = FakeConsentVerifier({"consent-1"})
    store = ProfileStore(verifier)

    store.register(profile)

    assert store.get(profile.profile_id) == profile
    verified_profile = store.assert_usable(profile.profile_id, "en")
    assert hasattr(profiles, "VerifiedVoiceProfile")
    assert isinstance(verified_profile, profiles.VerifiedVoiceProfile)
    assert verified_profile.profile == profile
    assert verifier.checked_profiles == [profile, profile]


def test_profile_store_requires_a_consent_verifier() -> None:
    with pytest.raises(TypeError):
        ProfileStore()


def test_missing_profile_is_rejected() -> None:
    store = make_store()

    with pytest.raises(ProfileNotFoundError):
        store.assert_usable("missing", "en")


def test_revoked_profile_is_rejected(tmp_path: Path) -> None:
    reference_audio = tmp_path / "reference.wav"
    reference_audio.write_bytes(b"audio")
    store = make_store()
    profile = make_profile(str(reference_audio), revoked=True)

    with pytest.raises(ProfileRevokedError):
        store.register(profile)

    with pytest.raises(ProfileNotFoundError):
        store.get(profile.profile_id)


@pytest.mark.parametrize("consent_id", ["", "   "])
def test_missing_or_empty_consent_is_rejected(tmp_path: Path, consent_id: str) -> None:
    reference_audio = tmp_path / "reference.wav"
    reference_audio.write_bytes(b"audio")

    with pytest.raises(ProfileError):
        make_store().register(make_profile(str(reference_audio), consent_id=consent_id))


@pytest.mark.parametrize(
    "consented_at",
    [
        datetime(2026, 7, 18),
        datetime.now(timezone.utc) + timedelta(minutes=1),
    ],
)
def test_invalid_consent_timestamp_is_rejected(
    tmp_path: Path, consented_at: datetime
) -> None:
    reference_audio = tmp_path / "reference.wav"
    reference_audio.write_bytes(b"audio")

    with pytest.raises(ProfileError):
        make_store().register(
            make_profile(str(reference_audio), consented_at=consented_at)
        )


def test_language_mismatch_is_rejected(tmp_path: Path) -> None:
    reference_audio = tmp_path / "reference.wav"
    reference_audio.write_bytes(b"audio")
    store = make_store()
    store.register(make_profile(str(reference_audio), language="fr"))

    with pytest.raises(ProfileLanguageMismatchError):
        store.assert_usable("profile-1", "en")


def test_missing_reference_audio_is_rejected() -> None:
    store = make_store()

    with pytest.raises(ReferenceAudioNotFoundError):
        store.register(make_profile("/does/not/exist.wav"))


def test_invented_consent_is_rejected_by_the_verifier(tmp_path: Path) -> None:
    reference_audio = tmp_path / "reference.wav"
    reference_audio.write_bytes(b"audio")
    profile = make_profile(str(reference_audio), consent_id="invented-consent")

    assert hasattr(profiles, "ConsentNotVerifiedError")
    with pytest.raises(profiles.ConsentNotVerifiedError):
        ProfileStore(FakeConsentVerifier(set())).register(profile)


def test_unavailable_consent_verifier_fails_closed(tmp_path: Path) -> None:
    reference_audio = tmp_path / "reference.wav"
    reference_audio.write_bytes(b"audio")

    assert hasattr(profiles, "ConsentNotVerifiedError")
    with pytest.raises(profiles.ConsentNotVerifiedError):
        ProfileStore(UnavailableConsentVerifier()).register(make_profile(str(reference_audio)))


def test_use_fails_closed_after_consent_is_invalidated(tmp_path: Path) -> None:
    reference_audio = tmp_path / "reference.wav"
    reference_audio.write_bytes(b"audio")
    profile = make_profile(str(reference_audio))
    verifier = MutableConsentVerifier()
    store = ProfileStore(verifier)
    store.register(profile)
    verifier.result = False

    with pytest.raises(profiles.ConsentNotVerifiedError):
        store.assert_usable(profile.profile_id, "en")

    assert verifier.checked_profiles == [profile, profile]


def test_persistent_store_rechecks_consent_at_use_after_restart(tmp_path: Path) -> None:
    reference_audio = tmp_path / "reference.wav"
    reference_audio.write_bytes(b"audio")
    database_path = tmp_path / "profiles.sqlite3"
    profile = make_profile(str(reference_audio))
    verifier = MutableConsentVerifier()

    repository = SQLiteProfileRepository(str(database_path))
    ProfileStore(verifier, repository).register(profile)
    repository.close()
    verifier.result = False

    restarted_repository = SQLiteProfileRepository(str(database_path))
    restarted_store = ProfileStore(verifier, restarted_repository)

    with pytest.raises(profiles.ConsentNotVerifiedError):
        restarted_store.assert_usable(profile.profile_id, "en")
    restarted_repository.close()


def test_truthy_non_boolean_consent_result_is_rejected_at_registration(
    tmp_path: Path,
) -> None:
    reference_audio = tmp_path / "reference.wav"
    reference_audio.write_bytes(b"audio")

    with pytest.raises(profiles.ConsentNotVerifiedError):
        ProfileStore(MutableConsentVerifier(result=1)).register(
            make_profile(str(reference_audio))
        )


def test_truthy_non_boolean_consent_result_is_rejected_at_use(tmp_path: Path) -> None:
    reference_audio = tmp_path / "reference.wav"
    reference_audio.write_bytes(b"audio")
    profile = make_profile(str(reference_audio))
    verifier = MutableConsentVerifier()
    store = ProfileStore(verifier)
    store.register(profile)
    verifier.result = "verified"

    with pytest.raises(profiles.ConsentNotVerifiedError):
        store.assert_usable(profile.profile_id, "en")


def test_signed_profile_round_trip_hides_reference_audio_from_client(tmp_path: Path) -> None:
    reference_audio = tmp_path / "reference.wav"
    reference_audio.write_bytes(b"audio")
    profile = make_profile(str(reference_audio))

    token = profile.to_signed_token(b"test-secret")
    restored = VoiceProfile.from_signed_token(token, b"test-secret")

    assert restored == profile
    assert str(reference_audio) not in profile.to_client_dict().values()
    assert "reference_audio_path" not in profile.to_client_dict()


def test_signed_profile_rejects_a_modified_token(tmp_path: Path) -> None:
    reference_audio = tmp_path / "reference.wav"
    reference_audio.write_bytes(b"audio")
    token = make_profile(str(reference_audio)).to_signed_token(b"test-secret")
    payload, signature = token.split(".")

    with pytest.raises(ValueError, match="invalid signed profile"):
        VoiceProfile.from_signed_token(f"{payload}x.{signature}", b"test-secret")
