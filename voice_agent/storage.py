"""SQLite persistence for consent-backed voice profiles."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from voice_agent.profiles import VoiceProfile


class SQLiteProfileRepository:
    """Persist profile metadata without retaining reference-audio content."""

    def __init__(self, database_path: str) -> None:
        self._database_path = database_path
        if database_path != ":memory:":
            Path(database_path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(database_path)
        self.initialize()

    def initialize(self) -> None:
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS voice_profiles (
                profile_id TEXT PRIMARY KEY,
                consent_id TEXT NOT NULL,
                consented_at TEXT NOT NULL,
                language TEXT NOT NULL,
                reference_audio_path TEXT NOT NULL,
                revoked INTEGER NOT NULL DEFAULT 0 CHECK (revoked IN (0, 1))
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS consent_records (
                consent_id TEXT PRIMARY KEY,
                language TEXT NOT NULL,
                recorded_at TEXT NOT NULL,
                revoked INTEGER NOT NULL DEFAULT 0 CHECK (revoked IN (0, 1))
            )
            """
        )
        self._connection.commit()

    def record_consent(self, consent_id: str, language: str, recorded_at: datetime) -> None:
        self._connection.execute(
            "INSERT OR REPLACE INTO consent_records(consent_id, language, recorded_at, revoked) VALUES (?, ?, ?, 0)",
            (consent_id, language, recorded_at.isoformat()),
        )
        self._connection.commit()

    def verify_consent(self, consent_id: str, language: str) -> bool:
        row = self._connection.execute(
            "SELECT language, revoked FROM consent_records WHERE consent_id = ?", (consent_id,)
        ).fetchone()
        return row is not None and row[0] == language and not bool(row[1])

    def save(self, profile: VoiceProfile) -> None:
        self._connection.execute(
            """
            INSERT INTO voice_profiles (
                profile_id, consent_id, consented_at, language, reference_audio_path, revoked
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(profile_id) DO UPDATE SET
                consent_id = excluded.consent_id,
                consented_at = excluded.consented_at,
                language = excluded.language,
                reference_audio_path = excluded.reference_audio_path,
                revoked = excluded.revoked
            """,
            (
                profile.profile_id,
                profile.consent_id,
                profile.consented_at.isoformat(),
                profile.language,
                profile.reference_audio_path,
                int(profile.revoked),
            ),
        )
        self._connection.commit()

    def get(self, profile_id: str) -> VoiceProfile | None:
        row = self._connection.execute(
            """
            SELECT profile_id, consent_id, consented_at, language, reference_audio_path, revoked
            FROM voice_profiles
            WHERE profile_id = ?
            """,
            (profile_id,),
        ).fetchone()
        if row is None:
            return None

        from voice_agent.profiles import VoiceProfile

        return VoiceProfile(
            profile_id=row[0],
            consent_id=row[1],
            consented_at=datetime.fromisoformat(row[2]),
            language=row[3],
            reference_audio_path=row[4],
            revoked=bool(row[5]),
        )

    def revoke(self, profile_id: str) -> None:
        self._connection.execute(
            "UPDATE voice_profiles SET revoked = 1 WHERE profile_id = ?", (profile_id,)
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()


class SQLiteConsentVerifier:
    def __init__(self, repository: SQLiteProfileRepository) -> None:
        self.repository = repository

    def verify(self, profile: VoiceProfile) -> bool:
        return self.repository.verify_consent(profile.consent_id, profile.language)
