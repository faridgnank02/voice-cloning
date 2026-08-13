"""Consent-backed voice profile storage and server-side serialization."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from voice_agent.storage import SQLiteProfileRepository


class ProfileError(ValueError):
    """Base class for profile authorization errors."""


class ProfileNotFoundError(ProfileError):
    """Raised when a requested voice profile does not exist."""


class ProfileRevokedError(ProfileError):
    """Raised when a profile's consent has been revoked."""


class ProfileLanguageMismatchError(ProfileError):
    """Raised when a profile is used with another language."""


class ReferenceAudioNotFoundError(ProfileError):
    """Raised when the server-side reference audio is unavailable."""


class ProfileConsentError(ProfileError):
    """Raised when a profile has no usable consent identifier."""


class InvalidConsentTimestampError(ProfileError):
    """Raised when a consent timestamp cannot establish valid consent."""


class UnverifiedProfileError(ProfileError):
    """Raised when code attempts to forge a TTS authorization capability."""


class ConsentNotVerifiedError(ProfileError):
    """Raised when independent consent verification is absent or unsuccessful."""


@dataclass(frozen=True)
class VoiceProfile:
    profile_id: str
    consent_id: str
    consented_at: datetime
    language: str
    reference_audio_path: str
    revoked: bool = False

    def to_signed_token(self, secret: bytes | str) -> str:
        """Serialize a profile for trusted server-side transport with HMAC integrity."""
        payload = json.dumps(
            {**asdict(self), "consented_at": self.consented_at.isoformat()},
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        encoded_payload = _encode(payload)
        signature = hmac.new(_secret_bytes(secret), payload, hashlib.sha256).digest()
        return f"{encoded_payload}.{_encode(signature)}"

    @classmethod
    def from_signed_token(cls, token: str, secret: bytes | str) -> VoiceProfile:
        """Restore a profile after verifying its HMAC signature."""
        try:
            encoded_payload, encoded_signature = token.split(".", maxsplit=1)
            payload = _decode(encoded_payload)
            signature = _decode(encoded_signature)
        except (ValueError, UnicodeEncodeError) as error:
            raise ValueError("invalid signed profile token") from error

        expected_signature = hmac.new(
            _secret_bytes(secret), payload, hashlib.sha256
        ).digest()
        if not hmac.compare_digest(signature, expected_signature):
            raise ValueError("invalid signed profile signature")

        try:
            values: dict[str, Any] = json.loads(payload)
            values["consented_at"] = datetime.fromisoformat(values["consented_at"])
            return cls(**values)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("invalid signed profile payload") from error

    def to_client_dict(self) -> dict[str, str | bool]:
        """Return public profile metadata without the private reference-audio path."""
        return {
            "profile_id": self.profile_id,
            "consent_id": self.consent_id,
            "consented_at": self.consented_at.isoformat(),
            "language": self.language,
            "revoked": self.revoked,
        }


@runtime_checkable
class ConsentVerifier(Protocol):
    """Trusted authority that validates consent for a complete voice profile."""

    def verify(self, profile: VoiceProfile) -> bool: ...


_VERIFICATION_MARKER = object()


@dataclass(frozen=True)
class VerifiedVoiceProfile:
    """A TTS capability issued only after the store authorizes a profile."""

    profile: VoiceProfile
    _verification: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._verification is not _VERIFICATION_MARKER:
            raise UnverifiedProfileError("voice profile has not been authorized for TTS")


class ProfileStore:
    """Profile store that validates profiles before their use by TTS."""

    def __init__(
        self,
        consent_verifier: ConsentVerifier,
        repository: SQLiteProfileRepository | None = None,
    ) -> None:
        self._consent_verifier = consent_verifier
        self._repository = repository
        self._profiles: dict[str, VoiceProfile] = {}

    def register(self, profile: VoiceProfile) -> None:
        self._assert_consent_is_valid(profile)
        self._assert_consent_is_verified(profile)
        if profile.revoked:
            raise ProfileRevokedError(f"profile {profile.profile_id!r} has been revoked")
        self._assert_reference_audio_exists(profile)
        self._profiles[profile.profile_id] = profile
        if self._repository is not None:
            self._repository.save(profile)

    def get(self, profile_id: str) -> VoiceProfile:
        if self._repository is not None:
            profile = self._repository.get(profile_id)
            if profile is None:
                raise ProfileNotFoundError(f"profile {profile_id!r} was not found")
            return profile
        try:
            return self._profiles[profile_id]
        except KeyError as error:
            raise ProfileNotFoundError(f"profile {profile_id!r} was not found") from error

    def assert_usable(self, profile_id: str, language: str) -> VerifiedVoiceProfile:
        profile = self.get(profile_id)
        self._assert_consent_is_valid(profile)
        self._assert_consent_is_verified(profile)
        if profile.revoked:
            raise ProfileRevokedError(f"profile {profile_id!r} has been revoked")
        if profile.language != language:
            raise ProfileLanguageMismatchError(
                f"profile {profile_id!r} is for {profile.language!r}, not {language!r}"
            )
        self._assert_reference_audio_exists(profile)
        return VerifiedVoiceProfile(profile, _VERIFICATION_MARKER)

    @staticmethod
    def _assert_consent_is_valid(profile: VoiceProfile) -> None:
        if not isinstance(profile.consent_id, str) or not profile.consent_id.strip():
            raise ProfileConsentError("profile consent_id is required")

        consented_at = profile.consented_at
        if (
            not isinstance(consented_at, datetime)
            or consented_at.tzinfo is None
            or consented_at.utcoffset() is None
            or consented_at > datetime.now(timezone.utc)
        ):
            raise InvalidConsentTimestampError(
                "profile consented_at must be an aware timestamp that is not in the future"
            )

    def _assert_consent_is_verified(self, profile: VoiceProfile) -> None:
        verifier = self._consent_verifier
        if not isinstance(verifier, ConsentVerifier):
            raise ConsentNotVerifiedError("a consent verifier is required to register profiles")
        try:
            is_verified = verifier.verify(profile)
        except Exception as error:
            raise ConsentNotVerifiedError("consent verification is unavailable") from error
        if is_verified is not True:
            raise ConsentNotVerifiedError("consent was not independently verified")

    @staticmethod
    def _assert_reference_audio_exists(profile: VoiceProfile) -> None:
        if not Path(profile.reference_audio_path).is_file():
            raise ReferenceAudioNotFoundError(
                f"reference audio for profile {profile.profile_id!r} is unavailable"
            )


def _secret_bytes(secret: bytes | str) -> bytes:
    return secret.encode("utf-8") if isinstance(secret, str) else secret


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)
